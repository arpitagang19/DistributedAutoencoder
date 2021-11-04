import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.layers.Dense(
            units=latent_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Orthogonal()
        )
        self.w1 = None

    def call(self, x):
        encoded = self.encoder(x)
        decoded = encoded @ tf.transpose(self.encoder.variables[0])
        return decoded

    def dist_train(self, models, input_split, W, learning_rate):
        num_nodes = W.shape[0]
        WW = (np.identity(num_nodes) + W) / 2
        WW1 = WW @ WW
        decoded = []
        w = []
        w1 = []
        curr_loss = []
        cov = []
        new_w = []
        for node in range(num_nodes):
            decoded.append(models[node](input_split[node]))
            w.append(models[node].encoder.variables[0])
            curr_loss.append((1 / input_split[node].shape[0]) * tf.norm(input_split[node] - decoded[node]))
            w1.append(models[node].w1)
            cov.append((1 / (input_split[node].shape[0])) * (input_split[node].transpose() @ input_split[node]))

        for node in range(num_nodes):
            if w1[node] is None:
                w_sum = 0
                for i in range(num_nodes):
                    w_sum += WW[node, i] * w[i]
                new_w.append(w_sum + learning_rate * (cov[node] @ w[node] - w[node] @ tf.linalg.band_part(
                    tf.transpose(w[node]) @ cov[node] @ w[node], 0, -1)))
                models[node].w1 = tf.Variable(initial_value=w[node], trainable=False)
            else:
                w_sum = 0
                w1_sum = 0
                for i in range(num_nodes):
                    w_sum += WW[node, i] * w[i]
                    w1_sum += WW1[node, i] * w1[i]
                new_w.append(2 * w_sum - w1_sum + learning_rate * (cov[node] @ w[node] - w[node] @ tf.linalg.band_part(
                    tf.transpose(w[node]) @ cov[node] @ w[node], 0, -1)) \
                             - learning_rate * (cov[node] @ w1[node] - w1[node] @ tf.linalg.band_part(
                    tf.transpose(w1[node]) @ cov[node] @ w1[node], 0, -1)))

        for node in range(num_nodes):
            if w1[node] is not None:
                w1[node].assign(w[node])
            w[node].assign(new_w[node])
        return tf.math.reduce_mean(curr_loss), w

    def dist_subspace(self, X, Y):
        X = X / np.linalg.norm(X, axis=0)
        Y = Y / np.linalg.norm(Y, axis=0)
        M = np.matmul(X.transpose(), Y)
        sine_angle = 1 - np.diag(M) ** 2
        dist = np.sum(sine_angle) / X.shape[1]
        return dist

    def fit_data(self, x_train_split, W, lr, epochs):
        models = []
        num_nodes = W.shape[0]
        for node in range(num_nodes):
            models.append(Autoencoder(self.latent_dim))
        losses = []
        angle_loss = []
        for epoch in range(epochs):
            curr_loss, w = self.dist_train(models, x_train_split, W, learning_rate=lr)
            losses.append(curr_loss)
            err = 0
            for i in range(n):
                err += self.dist_subspace(w[i].numpy(), Eigvec)
            angle_loss.append(err / n)
        return losses, angle_loss