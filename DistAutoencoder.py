import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import networkx as nx
import pickle

class DistAutoencoder(Model):
    def __init__(self, latent_dim, comm):
        super(DistAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.layers.Dense(
            units=latent_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Orthogonal()
        )
        self.w1 = None
        self.comm = comm
        self.n = self.comm.Get_size()
        self.p = 0.5
        self.curr_node = comm.Get_rank()
        self.W = None
        self.type = 'erdos-renyi'
        if self.curr_node == 0:
            self.W = self.createGraph()
        self.W = comm.bcast(self.W)
        self.WW = (np.identity(self.n) + self.W) / 2
        self.WW1 = self.WW @ self.WW


    def call(self, x):
        encoded = self.encoder(x)
        decoded = encoded @ tf.transpose(self.encoder.variables[0])
        return decoded

    def dist_train(self, input_split, epoch, Eigvec, learning_rate):
        decoded = self.call(input_split)
        w = self.encoder.variables[0]
        curr_loss = (1 / input_split.shape[0]) * tf.norm(input_split - decoded)
        err = self.dist_subspace(w.numpy(), Eigvec)
        w1 = self.w1
        cov = (1 / (input_split.shape[0])) * (input_split.transpose() @ input_split)
        for i in range(self.n):
            if self.W[self.curr_node, i]:
                self.comm.send([w, w1], dest=i, tag=epoch)
        if w1 is None:
            w_sum = 0
            for i in range(self.n):
                if self.W[self.curr_node, i]:
                    w_sum += self.WW[self.curr_node, i] * self.comm.recv(source=i, tag=epoch)[0]
            new_w = w_sum + learning_rate * (cov @ w - w @ tf.linalg.band_part(tf.transpose(w) @ cov @ w, 0, -1))
            self.w1 = tf.Variable(initial_value=w, trainable=False)
        else:
            w_sum = 0
            w1_sum = 0
            for i in range(self.n):
                if self.W[self.curr_node, i]:
                    wi, w1i = self.comm.recv(source=i, tag= epoch)
                    w_sum += self.WW[self.curr_node, i] * wi
                    w1_sum += self.WW1[self.curr_node, i] * w1i
            new_w = 2 * w_sum - w1_sum + learning_rate * (cov @ w - w @ tf.linalg.band_part(tf.transpose(w) @ cov @ w, 0, -1)) \
                         - learning_rate * (cov @ w1 - w1 @ tf.linalg.band_part(tf.transpose(w1) @ cov @ w1, 0, -1))
        if w1:
            w1.assign(w)
        w.assign(new_w)
        total_loss = self.comm.Allreduce(curr_loss)
        total_err = self.comm.Allreduce(err)
        return total_loss/self.n, total_err/self.n

    def dist_subspace(self, X, Y):
        X = X / np.linalg.norm(X, axis=0)
        Y = Y / np.linalg.norm(Y, axis=0)
        M = np.matmul(X.transpose(), Y)
        sine_angle = 1 - np.diag(M) ** 2
        dist = np.sum(sine_angle) / X.shape[1]
        return dist

    def fit_data(self, lr, epochs):
        losses = []
        angle_loss = []
        x_train_split = pickle.load(open("TrainingData/x_train{}.pickle".format(self.curr_node)))
        Eigvec = pickle.load(open("TrainingData/X_gt.pickle"))
        for epoch in range(epochs):
            curr_loss, err = self.dist_train(x_train_split, epoch, Eigvec, learning_rate=lr)
            losses.append(curr_loss)
            angle_loss.append(err)
        return losses, angle_loss


    def createGraph(self):
        conn = False
        G = None
        if self.type == 'erdos-renyi':
            while not conn:
                G = nx.erdos_renyi_graph(self.n, self.p)
                conn = nx.is_connected(G)
        if self.type == 'cycle':
            while not conn:
                G = nx.cycle_graph(self.n)
                conn = nx.is_connected(G)
        if self.type == 'expander':
            while not conn:
                G = nx.margulis_gabber_galil_graph(self.n)
                conn = nx.is_connected(G)
        if self.type == 'star':
            while not conn:
                G = nx.star_graph(self.n - 1)
                conn = nx.is_connected(G)
        self.G = G
        self.create_weight_matrix_metropolis(G)
        return self.W

    def create_weight_matrix_metropolis(self, G):
        A = nx.to_numpy_matrix(G)
        # degree of the nodes
        D = np.sum(A, 1)
        D = np.array(D.transpose())
        D = D[0, :]
        D = D.astype(np.int64)
        self.W = np.zeros((A.shape))
        for i in range(0, A.shape[0]):
            for j in range(i, A.shape[1]):
                if A[i, j] != 0 and i != j:
                    self.W[i, j] = 1 / (max(D[i], D[j])+1)
                    self.W[j, i] = self.W[i, j]
        for i in range(0, A.shape[0]):
            for j in range(i, A.shape[1]):
                if i == j:
                    self.W[i, j] = 1 - np.sum(self.W[i, :])