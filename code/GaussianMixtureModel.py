import numpy as np
import random as rnd
import scipy.stats as stats
import operator
from sklearn.cluster import KMeans
from collections import Counter

class GaussianMixtureModel:
    def __init__(self, n_components=1, max_iter=100, tolerance=1e-3):
        self.n_components = n_components # # of components in GMM
        self.max_iter = max_iter # max iteration of EM
        self.tol = tolerance # convergence threshold of EM
        self.priors = list() # prior prob of each gaussian distribution in GMM
        self.gaussians = list() # mean and covariance of each gaussian distribution in GMM
        self.logsum = 0 # log likelihood for GMM parameter setting
        self.cmp = 1e-8 # compensate for denominator in zero division (precision)

    def fit(self, train_data):
        # initialize the parameters of GMM
        # self.priors = self.initialize_prior()
        # self.gaussians = self.initialize_gaussian_distribution(train_data)
        self.initialize_all_parameters(train_data)
        #print self.priors # debug
        #print self.gaussians # debug
        train_size = len(train_data)
        p_ij = np.matrix([[0]*self.n_components]*train_size, dtype=np.float64) # membership prob
        iter_count = 0
        while (True): # repeat E-M until convergence
            iter_count += 1
            self.logsum = np.sum(map(self.log_sum, train_data))
            #print iter_count, self.logsum # debug
            # E-step: estimate the conditional distribution of membership given data
            for i in range(train_size):
                x_i = train_data[i]
                prob_sum = 0
                for j in range(self.n_components):
                    mu = self.gaussians[j][0]
                    tho = self.gaussians[j][1]
                    p_ij[i, j] = self.priors[j] * stats.norm(mu, tho).pdf(x_i) # prob of component j
                    prob_sum += p_ij[i, j]
                for j in range(self.n_components):
                    p_ij[i, j] /= (prob_sum+self.cmp) # normalize for membership prob
            # M-step: estimate parameters of GMM
            delta_mu, delta_tho, delta_prior = 0, 0, 0 # change of parameters against last iteration
            total_p_ij = np.sum(p_ij)
            for j in range(self.n_components):
                p_j = np.sum(p_ij[:, j])
                prior = p_j / (total_p_ij+self.cmp) # estimate prior
                delta_prior += abs(self.priors[j] - prior)
                self.priors[j] = prior
                weights = np.hstack(p_ij[:,j]).tolist()[0]
                #print weights # debug
                mu = np.sum(map(lambda x,p: x*p, train_data, weights)) / (p_j+self.cmp) # estimate mean
                tho = np.sqrt(np.sum(map(lambda x,p: p*(x-mu)**2, train_data, weights)) / (p_j+self.cmp)) # estimate covariance
                delta_mu += abs(self.gaussians[j][0] - mu)
                delta_tho += abs(self.gaussians[j][1] - tho)
                #print j, mu, tho # debug
                self.gaussians[j][0] = mu
                self.gaussians[j][1] = (tho+self.cmp)
            # check convergence
            logsum = np.sum(map(self.log_sum, train_data))
            if abs(self.logsum-logsum)<self.tol or iter_count>=self.max_iter:
                break
            self.logsum = logsum

    def predict(self, test_data):
        labels = list()
        for data in test_data:
            posteriors = map(lambda pr,gs: pr*stats.norm(gs[0], gs[1]).pdf(data), self.priors, self.gaussians)
            #print posteriors # debug
            index, _ = max(enumerate(posteriors), key=operator.itemgetter(1))
            labels.append(index)
        return labels

    def initialize_all_parameters(self, train_data):
        kmeans = KMeans(n_clusters=self.n_components).fit(train_data)
        labels = kmeans.predict(train_data)
        counts = Counter(labels)
        centroids = np.hstack(kmeans.cluster_centers_)
        data_cluster_pairs = zip(np.hstack(train_data), labels)
        size_of_data = len(labels)
        num_of_clusters = len(centroids)
        gaussians = np.matrix([[0,0]]*num_of_clusters, dtype=np.float64)
        priors = np.array([0] * num_of_clusters, dtype=np.float64)
        for dc in data_cluster_pairs:
            #print dc # debug
            gaussians[dc[1], 1] += (dc[0] - centroids[dc[1]])**2
        #print type(gaussians) # debug
        for i, ct in enumerate(centroids):
            #print i, ct # debug
            gaussians[i, 0] = ct
            gaussians[i, 1] = np.sqrt(gaussians[i, 1]/(counts[i]+self.cmp))
            priors[i] = float(counts[i])/size_of_data
        self.gaussians = gaussians.tolist()
        self.priors = priors.tolist()

    def log_sum(self, x):
        return np.sum(map(lambda pr,gs: pr*stats.norm(gs[0], np.sqrt(gs[1])).pdf(x), self.priors, self.gaussians))
