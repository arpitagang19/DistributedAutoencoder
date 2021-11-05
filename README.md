# DistributedAutoencoder
## General Information

This repository contains implementation of a distributed training system for autoencoder in case where data samples are distributed across an arbitrary network like the IoT. The training loop is based on the Hebbian rule such that the encoder weights converge to the eigenvectors of the input correlation matrix.


## Summary of Experiments

This set of codes use tensorflow for implementation of the autoencoder training system and MPI for simulating the distributed network. Message Passing Interface or MPI is a standardized and portable message-passing standard designed to function on parallel computing architectures. The number of nodes in the simulated network is equal to the number of available cores/CPUs/GPUs.

## Data

Data is generated in the Data.py file. It takes in the following arguments: d, N, eigen_gap, K

1. dimension (d): Dimension of each sample.
2. Number of samples (N): Number of samples generated
3. Dimension of encoded samples (K) = Number of eigenvectors to be estimated
4. eigen_gap: Eigen gap between the Kth and (K+1)th eigenvalues of the input covariance matrix.

## Summary of Code

The main driving script is ```test.py```. The necessary functions required for the experiments are called within this functions and all the necessary parameters like number of eigenvectors to be estimated `K` , number of nodes in the network `num_nodes` etc. are given as input to this python file during execution. 

```
usage: test.py [-h] [-d DIMENSION] [-K K] [-EG EIGENGAP] [-lr LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  -d DIMENSION, --dimension DIMENSION
                        Dimension of the data samples, default value is 20
  -K K, --K K           number of eigenvectors to be estimated, default number
                        is 5
  -EG EIGENGAP, --eigengap EIGENGAP
                        eigengap between Kth and (K+1)th eigenvalues
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate, default value is 0.1
```

## Requirements and Dependencies

The code is written in Python. To reproduce the environment with necessary dependencies needed for running of the code in this repo, we recommend that the users create a `conda` environment using the `environment.yml` YAML file that is provided in the repo. Assuming the conda management system is installed on the user's system, this can be done using the following: 

``` $ conda env create -f environment.yml ```

In the case users don't have conda installed on their system, they should check out the `environment.yml` file for the appropriate version of Python as well as the necessary dependencies with their respective versions needed to run the code in the repo.

## Contributors

The algorithmic implementations, experiments and reproduciblity of these codes was done by: [Arpita Gang](https://www.linkedin.com/in/arpitagang/)








