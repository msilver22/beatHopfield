import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

"""
    Creation Hopfield's model
"""

class HopfieldNetwork(object):      
    def train_weights(self, train_data):
        print("Start to train weights...")
        self.num_neuron = train_data.shape[1]
        self.num_patterns = train_data.shape[0]
        self.patterns = train_data
        
        #initialize weights
        J = np.zeros((self.num_neuron, self.num_neuron))
        
        # Hebbian rule
        for i in tqdm(range(0,self.num_neuron)):
            for j in range(i + 1, self.num_neuron):
                for mu in range(0, self.num_patterns):
                    J[i, j] += train_data[mu, i] * train_data[mu, j]

        J = (J + J.T) / self.num_neuron
        self.J = J 
    
    def predict(self, test_data, temperature):
        
        sigma = test_data.copy().T
        sigma = sigma.T
        #parameters
        N=self.num_neuron 
        K=self.num_patterns 
        alpha = K/N
        """print("The load of the model is", alpha)"""
        """print("Start to predict...")"""
        """print("The temperature is ", temperature)"""
        T = temperature
        beta = 1.0 / T

        MCstat_step=50
        MCrelax_step=1

        # Inizializzo la magnetizzazione
        magn_mattis_matrix = np.zeros((self.num_patterns,MCstat_step))

        #MONTECARLO simulations
        for stat in range(0, MCstat_step):
            for step in range(0,MCrelax_step):
                #montecarlo step
                for i in range(0,N):
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
    
    def predict_roughWay(self,test_data):
        """print("Start to predict...")"""

        s = test_data.copy()
        #PARALLEL
        """for i in range(0,2):
            s = np.sign(self.J @ s)   """
        #SEQUENTIAL
        num_iter=10*self.num_neuron
        magn_mattis_matrix = np.zeros((self.num_patterns,num_iter))
        for i in range(num_iter):
            k = np.random.randint(0, self.num_neuron)
            s[k] = np.sign(np.dot(self.J[k,:],s))
            for mu in range(0,self.num_patterns):
                magn_mattis_matrix[mu,i]= np.dot(s,self.patterns[mu,:])/self.num_neuron 
        
        return s, magn_mattis_matrix
    

    
    def get_corrupted(self, pattern,r):
        sample_size = int(self.num_neuron*r)
        I = np.random.choice(len(pattern),size = sample_size, replace=False)
        corrupted = pattern.copy()
        for i in range(len(I)):
            corrupted[I[i]] = -1*corrupted[I[i]]

        return corrupted
    

    def pattern_similarity(self, patterns, pattern_test):
        iter = patterns.shape[0]
        comparisons = np.zeros(iter)
        for i in range(iter):
            correct_components = sum(x == y for x, y in zip(patterns[i, :], pattern_test))
            similarity = correct_components / len(pattern_test)
            comparisons[i] = similarity

        """print("Percentage of equal components wrt stored patterns")
        print(comparisons)"""
        prediction = np.argmax(comparisons)
        return prediction
    
        
