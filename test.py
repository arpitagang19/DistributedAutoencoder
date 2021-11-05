import numpy as np
from Data import Data
from DistAutoencoder import DistAutoencoder
import os
import pickle
from mpi4py import MPI
import argparse


comm = MPI.COMM_WORLD
num_nodes = comm.Get_size()
# initialize variables
parser = argparse.ArgumentParser()
parser.add_argument("-d","--dimension", help="Dimension of the data samples, default value is 20", type=int, default=20)
parser.add_argument("-K", "--K", help="number of eigenvectors to be estimated, default number is 5", type = int, default=5)
parser.add_argument("-EG","--eigengap", help="eigengap between Kth and (K+1)th eigenvalues", type = float, default=0.8)
parser.add_argument("-lr", "--learning_rate", help="learning rate, default value is 0.1", type = float, default=0.1)
parser.add_argument("-N", "--num_samples", help="learning rate, default value is 100000", type = int, default=10000)
args = parser.parse_args()

N = args.num_samples                       # number of data samples

d = args.dimension                  # dimension of data samples
K = args.K                          # number of eigenvectors to be estimated
eigengap = args.eigengap            # eigen gap between K+1 and Kth eigenvalue
lr = args.learning_rate             # initial step size for DSA
type = 1                            # type = 1: eigenvalues of data covariance matrix are distinct,
                                                # 2: repeated eigenvalues of data covariance matrix
# generate synthetic data
test_data = Data("synthetic")
x_train = test_data.generateSynthetic(d, N, eigengap, K, type)

if not os.path.exists("TrainingData"):
    os.mkdir("TrainingData")
x_train_split = np.vsplit(x_train, num_nodes)
for i in range(num_nodes):
    pickle.dump(x_train_split[i], open('TrainingData/x_train{}.pickle'.format(i), 'wb'))
X_gt = test_data.computeTrueEV(x_train, K)
pickle.dump(X_gt, open('TrainingData/X_gt.pickle', 'wb'))

# run algorithms on the data
dist_auto = DistAutoencoder(K, comm)
losses, angle_loss = dist_auto.fit_data(lr, epochs=500)


