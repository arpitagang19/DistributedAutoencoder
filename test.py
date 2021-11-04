import numpy as np
from GraphTopology import GraphType
from Data import Data
from DistAutoencoder import DistAutoencoder
import os
import pickle
from mpi4py import MPI

comm = MPI.COMM_WORLD
num_nodes = comm.Get_size()
# initialize variables
N = 100000       # number of data samples

d = 20           # dimension of data samples
K = 5            # number of eigenvectors to be estimated
eigengap = 0.8  # eigen gap between K+1 and Kth eigenvalue

gtype = 'erdos-renyi'   # type of graph: erdos-renyi, cycle, star
# p = 0.5                 # connectivity for erdos renyi graph
lr = 0.1        # initial step size for DSA
type = 1



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


