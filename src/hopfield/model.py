from tqdm import tqdm
import random
import numpy as np
import os

class HopfieldNetwork(object):
    def __init__(self, patterns):

        self.patterns = patterns
        self.num_patterns = patterns.shape[0]
        self.num_neurons = patterns.shape[1]

        weights_path = './src/hopfield/weights.npy'
        if os.path.exists(weights_path):
            print("Loading weights from file.")
            J = np.load(weights_path)
        else:
            print("Training weights.")
            J = self.train_weights(patterns)
        self.J = J

        if np.any(self.J == None):
            raise ValueError("The weight matrix J contains None elements.")
        else: 
            print("Hopfield network initialized successfully.")
        
    
    def train_weights(self, train_data):        
        
        J = np.zeros((self.num_neurons, self.num_neurons))
        # Hebbian rule
        for i in range(0,self.num_neurons):
            for j in range(i + 1, self.num_neurons):
                for mu in range(0, self.num_patterns):
                    J[i, j] += train_data[mu, i] * train_data[mu, j]

        J = (J + J.T) / self.num_neurons
        self.J = J
        #np.save('./src/hopfield/weights.npy', self.J)
        return J

    def predict(self, test_data, temperature):
        sigma = test_data.copy()
        #parameters
        N=self.num_neurons
        K=self.num_patterns
        beta = 1.0 / temperature
        MCstat_step=100
        MCrelax_step=10
        # Inizialize the magnetization
        magn_mattis_matrix = np.zeros((self.num_patterns,MCstat_step))

        #MONTECARLO simulations
        for stat in range(0, MCstat_step):
            for _ in range(0,MCrelax_step):
                #montecarlo step
                for _ in range(0,N):
                    k = np.random.randint(0, N)  #flipping candidate
                    deltaE=2*sigma[k]*np.dot(sigma,self.J[:,k]) #if deltaE<0 always accept
                    ratio=np.exp(-beta*deltaE)
                    gamma=np.minimum(ratio,1)
                    if np.any(random.uniform(0,1) < gamma):
                        sigma[k] = -sigma[k]  #flipping
            for mu in range(0,self.num_patterns):
                magn_mattis_matrix[mu,stat]= np.dot(sigma,self.patterns[mu,:])/N

        predicted = sigma
        return predicted,magn_mattis_matrix

    def get_corrupted(self, pattern,r):
        sample_size = int(self.num_neurons*r)
        I = np.random.choice(len(pattern),size = sample_size, replace=False)
        corrupted = pattern.copy()
        for i in range(len(I)):
            corrupted[I[i]] *= -1

        return corrupted
