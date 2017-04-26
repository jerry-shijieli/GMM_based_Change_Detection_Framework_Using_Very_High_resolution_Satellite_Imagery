# Script built based on codes of ipython notebook v3

import numpy as np
import os
import random
import matplotlib.pyplot as plt

from skimage import io
from matplotlib import colors
from matplotlib import cm
from GaussianMixtureModel import GaussianMixtureModel


# divide image arrays into grids based on preset step size
# @param: 2D RGB image as ndarray
# @param: grid_step_size should be divisible to image size
# @return: list of lists of divided ndarray (image grids)
def divide_image_into_grids(image, grid_step_size=50):
    assert image.shape[0]%grid_step==0
    assert image.shape[1]%grid_step==0
    image_grids = list()
    grid_rows = image.shape[0]/grid_step # number of grids along rows
    grid_cols = image.shape[1]/grid_step # number of grids along columns
    for r in range(grid_rows):
        row_image_grids = list()
        for c in range(grid_cols):
            tmp = image[r*grid_step:(r+1)*grid_step-1, c*grid_step:(c+1)*grid_step-1, :]
            row_image_grids.append(tmp)
        image_grids.append(row_image_grids)
    return image_grids

# approximate an image by a Gaussian distribution
# @param image matrix
# @return mean and variance matrix for this gaussian distribution
def image_to_gaussian_distribution(image):
    # convert image pixels into list of vectors
    data = list() # convert each pixel info into a vector
    rows, cols, channels = image.shape
    for r in range(rows):
        vecs = image[r]
        data.extend(vecs)
    # fit data into a gaussian distribution
    data_mean = np.mean(data, axis=0) # mean vector of gaussian distribution
    data_cov = np.cov(data, rowvar=False) # covariance matrix of gaussian distribution
    return (data_mean, data_cov)

# approximate an image by a Gaussian distribution including pixel coordinates
# @param image matrix
# @return mean and variance matrix for this gaussian distribution
def image_to_gaussian_distribution_withxy(image):
    # convert image pixels into list of vectors
    data = list() # convert each pixel info into a vector
    rows, cols, channels = image.shape
    for r in range(rows):
        index = zip([r]*cols, range(cols))
        img_info = zip(index, image[r])
        vecs = map(lambda x: np.concatenate((np.array(x[0]), x[1]),axis=0) , img_info)
        data.extend(vecs)
    # fit data into a gaussian distribution
    data_mean = np.mean(data, axis=0) # mean vector of gaussian distribution
    data_cov = np.cov(data, rowvar=False) # covariance matrix of gaussian distribution
    return (data_mean, data_cov)

# compute the Kullback-Leibler(KL) divergence between two Gaussian distributions
# @param mean and covariance matrix of two gaussian distributions
# @return KL divergence (asymmetric)
def gaussian_KL_divergence(gsd_1, gsd_2):
    gp_mean, gp_cov = gsd_1[0], gsd_1[1] # mean and covariance of gaussian distr
    gq_mean, gq_cov = gsd_2[0], gsd_2[1]
    term1 = np.log(np.linalg.det(gp_cov) / np.linalg.det(gq_cov))
    mat_inv_q = np.linalg.inv(gq_cov)
    term2 = np.trace(np.matmul(mat_inv_q, gp_cov))
    diff_mean_pq = gp_mean-gq_mean
    term3 = np.matmul(diff_mean_pq, np.matmul(mat_inv_q, np.matrix.transpose(diff_mean_pq)))
    KL_div = (term1 + term2 + term3) / 2
    return KL_div

# @return KL divergence (symmetric)
def symmetric_KL_divergence(gsd_1, gsd_2):
    return (gaussian_KL_divergence(gsd_1, gsd_2)+gaussian_KL_divergence(gsd_2, gsd_1)) / 2


if __name__ == '__main__':
    # read in image data as n-dimension array
    imgdir = '../dataset'
    imgfile1 = os.path.join(imgdir, 'k02-05m-cropped.png')
    imgfile2 = os.path.join(imgdir, 'k12-05m-cropped.png')
    img1 = io.imread(imgfile1)
    img2 = io.imread(imgfile2)

    # display and save the comparison of cropped satellite images
    fig = plt.figure(figsize=(36,18))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(img1)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(img2)
    fig.suptitle("Comparison of cropped temporal satellite images")
    fig.savefig("Comparison_of_cropped_temporal_satellite_images.png")

    # paraemeter setting to divide image into square grids
    grid_step = 140  # unit in pixel, should be divisible to the image size
    image_grids_1 = divide_image_into_grids(img1, grid_step)
    image_grids_2 = divide_image_into_grids(img2, grid_step)

    # convert each grid into a gaussian distribution, save gaussian parameters into a list
    num_of_rows = len(image_grids_1)
    num_of_cols = len(image_grids_1[0])
    image_gaussians_1 = list()
    for r in range(num_of_rows):
        image_gaussians_1.append(map(image_to_gaussian_distribution_withxy, image_grids_1[r]))
    image_gaussians_2 = list()
    for r in range(num_of_rows):
        image_gaussians_2.append(map(image_to_gaussian_distribution_withxy, image_grids_2[r]))

    # generate KL divergence map based on Gaussian distribution of each grid
    KL_div_map = list()
    for r in range(num_of_rows):
        KL_div_map.append(map(symmetric_KL_divergence, image_gaussians_1[r], image_gaussians_2[r]))

    # display and save the KL divergence map
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111)
    cax = ax.imshow(KL_div_map, cmap=plt.get_cmap('brg'), vmin=0, vmax=50, interpolation='none')
    # cax = ax.imshow(KL_div_map, cmap=cm.coolwarm, vmin=0, vmax=50, interpolation='none')
    cbar = fig.colorbar(cax, ticks=[0, 50])
    cbar.ax.set_yticklabels(['0', '50'])
    fig.suptitle("KL divergence map")
    fig.savefig("KL_divergence_map.png")

    # plot KL map overlaid on satellite map for comparison of changes
    fig = plt.figure(figsize=(36, 18))
    extent = 0, img2.shape[0], 0, img2.shape[1]
    plt.subplot(121).imshow(img1, extent=extent)
    plt.subplot(121).imshow(KL_div_map, cmap=plt.get_cmap('brg'), vmin=0, vmax=50, interpolation='none', alpha=0.3, extent=extent)
    plt.subplot(122).imshow(img2, extent=extent)
    plt.subplot(122).imshow(KL_div_map, cmap=plt.get_cmap('brg'), vmin=0, vmax=50, interpolation='none', alpha=0.3, extent=extent)
    fig.suptitle("KL divergence maps for change detection")
    fig.savefig("KL_divergence_maps_for_change_detection.png")

    # Build GMM model-based clustering model by subset of KL divergence map
    KL_values = np.vstack(np.ravel(KL_div_map))  # flatten the 2D matrix to 1D array, then transpose into column
    sample_ratio = 0.6  # subset ratio in full data set
    KL_samples = random.sample(KL_values, int(sample_ratio * len(KL_values)))  # select subset
    K = 4  # number of gaussian mixtures
    # from sklearn import mixture
    # gmm = mixture.GaussianMixture(n_components=K)
    gmm = GaussianMixtureModel(n_components=K, max_iter=20, tolerance=1e-2)
    gmm.fit(KL_samples)  # training GMM model

    # Apply GMM clustering model to categorize the changes on KL divergence map
    KL_change_map = list()
    for KL_div_row in KL_div_map:
        KL_change_map.append(gmm.predict(np.vstack(KL_div_row)))

    # Plot the change map to detect changes
    fig = plt.figure(figsize=(30, 30))
    ax = plt.subplot(111)
    #cmap = colors.LinearSegmentedColormap.from_list('segmented', ['#3366cc', '#33cc33', '#ffcc00', '#ff0000'], N=K)
    cmap1 = colors.LinearSegmentedColormap.from_list(11, colors=['MediumPurple', 'SpringGreen', 'Yellow', 'OrangeRed'], N=K)
    cax = ax.imshow(KL_change_map, cmap=cmap1, interpolation='none', alpha=1.0)
    cbar = fig.colorbar(cax, ticks=[i for i in range(K)])
    cbar.ax.set_yticklabels([str(i) for i in range(K)])
    fig.suptitle("Temporal change clustering by Gaussian Mixture Model")
    fig.savefig("Temporal_change_clustering_by_Gaussian_Mixture_Model.png")

    # overlaid GMM change cluster map on satellite maps for comparison and evaluation
    fig = plt.figure(figsize=(36, 18))
    cmap1 = colors.LinearSegmentedColormap.from_list(11, colors=['MediumPurple', 'SpringGreen', 'Yellow', 'OrangeRed'], N=K)
    extent = 0, img2.shape[0], 0, img2.shape[1]
    plt.subplot(121).imshow(img1, extent=extent)
    plt.subplot(121).imshow(KL_change_map, cmap=cmap1, interpolation='none', alpha=0.3, extent=extent)
    plt.subplot(122).imshow(img2, extent=extent)
    plt.subplot(122).imshow(KL_change_map, cmap=cmap1, interpolation='none', alpha=0.3, extent=extent)
    fig.suptitle("Change detection maps by GMM for evaluation")
    fig.savefig("Change_detection_maps_by_GMM_for_evaluation.png")