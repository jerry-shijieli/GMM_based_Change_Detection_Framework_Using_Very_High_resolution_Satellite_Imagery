import numpy as np
import random as rnd
import scipy.stats as stats


class GaussianMixtureModel:
    def __init__(self, n_components=1, max_iter=100, tolerance=1e-3):
        self.n_components = n_components # # of components in GMM
        self.max_iter = max_iter # max iteration of EM
        self.tol = tolerance # convergence threshold of EM
        self.priors = list() # prior prob of each gaussian distribution in GMM
        self.gaussians = list() # mean and covariance of each gaussian distribution in GMM

    def fit(self, train_data):
        # initialize the parameters of GMM
        self.priors = self.initialize_prior()
        self.gaussians = self.initialize_gaussian_distribution(train_data)
        train_size = len(train_data)
        p_ij = np.matrix([[0]*self.n_components]*train_size, dtype=np.float64) # membership prob
        iter_count = 0
        while (True): # repeat E-M until convergence
            iter_count += 1
            # E-step: estimate the conditional distribution of membership given data
            for i in range(train_size):
                x_i = train_data[i]
                prob_sum = 0
                for j in range(self.n_components):
                    mu = self.guassians[j][0]
                    tho = self.gaussians[j][1]
                    p_ij[i, j] = self.priors[j] * stats.norm(mu, tho).pdf(x_i) # prob of component j
                    prob_sum += p_ij[i, j]
                for j in range(self.n_components):
                    p_ij[i, j] /= prob_sum # normalize for membership prob
            # M-step: estimate parameters of GMM
            delta_mu, delta_tho, delta_prior = 0, 0, 0 # change of parameters against last iteration
            total_p_ij = np.sum(p_ij)
            for j in range(self.n_components):
                p_j = np.sum(p_ij[:, j])
                prior = p_j / total_p_ij # estimate prior
                delta_prior += abs(self.priors[j] - prior)
                self.priors[j] = prior
                mu = np.sum(map(lambda x,p: x*p, train_data, np.hstack(p_ij[:,j]))) / p_j # estimate mean
                tho = np.sum(map(lambda x,p: p*(x-mu)**2, train_data, np.hstack(p_ij[:,j]))) / p_j # estimate covariance
                delta_mu += abs(self.gaussians[j][0] - mu)
                delta_tho += abs(self.gaussians[j][1] - tho)
                self.gaussians[j][0] = mu
                self.gaussians[j][1] = tho
            # check convergence
            if max([delta_prior, delta_mu, delta_tho])<self.tol or iter_count>=max_iter:
                break

    def predict(self, test_data):
        pass

    def initialize_prior(self):
        priors = list()
        remainder = 1.0
        for _ in range(self.n_components-1):
            prb = rnd.random()*remainder # prob or weight of each gaussian components
            priors.append(prb)
            remainder -= prb
        priors.append(remainder)
        return priors

    def initialize_gaussian_distribution(self, train_data):
        minVal, maxVal = min(train_data), max(train_data)
        gaussians = list()
        interval = maxVal - minVal
        for _ in range(self.n_components):
            mu = rnd.random()*interval + minVal # mean
            tho = rnd.random()*interval # variance
            gaussians.append([mu, tho])
        return gaussians