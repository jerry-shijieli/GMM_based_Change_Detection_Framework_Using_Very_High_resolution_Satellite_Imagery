import numpy as np
import random as rnd
from GaussianMixtureModel import GaussianMixtureModel
from sklearn import mixture

K = 4

gmm = GaussianMixtureModel(n_components=K)
data = np.vstack([rnd.random() for i in range(50)])
gmm.fit(data)
print gmm.priors
print gmm.gaussians
print np.array(gmm.predict(data))

gmmx = mixture.GaussianMixture(n_components=K, covariance_type='spherical')
gmmx.fit(data)
print gmmx.predict(data)
print gmmx.weights_
print np.hstack(gmmx.means_)