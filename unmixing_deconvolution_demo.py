#!/usr/bin/env python

# Richardson-Lucy Deconvolution Spectral Unmixing demonstration
# James Manton, 2022
# jmanton@mrc-lmb.cam.ac.uk


import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from pandas import read_csv
 
num_iters = 100

def H(x, M, sigmas):
    blurreds = np.zeros((M.shape[0], M.shape[1], 64, 64))
    for o in range(M.shape[1]):
        for c in range(M.shape[0]):
            blurreds[c, o, :, :] = gaussian_filter(x[o, :, :], sigmas[c], mode='constant')
    ims = np.zeros((M.shape[0], 64, 64))
    for c in range(M.shape[0]):
        for o in range(M.shape[1]):
            ims[c, :, :] = ims[c, :, :] + M[c, o] * blurreds[c, o]
    return ims

def HT(x, M, sigmas):
    splatted = np.zeros((M.shape[0], M.shape[1], 64, 64))
    for o in range(M.shape[1]):
        for c in range(M.shape[0]):
            splatted[c, o, :, :] = gaussian_filter(x[c, :, :] * M[c, o], sigmas[c], mode='constant')
    ests = np.zeros((M.shape[1], 64, 64))
    for o in range(M.shape[1]):
        ests[o, :, :] = np.mean(splatted[:, o, :, :], axis=0)
    return ests


# Generate mixed, blurred data
letters = tifffile.imread('demo_ground_truth.tif')
M = read_csv('demo_PRISM_mixing_matrix_EGFP_Qdot_525_Qdot_565_Qdot_585_Qdot_605_Qdot_655_Qdot_705_TagBFP.csv', header=0)
M = np.array(M, dtype=np.float32)
sigmas = np.array([491, 514, 532, 561, 594, 633, 670, 710]) / 491

print(M)
print(np.linalg.pinv(M))

ims = H(letters, M, sigmas)
ims = np.random.poisson(ims)
tifffile.imwrite('demo_raw_data.tif', ims)

est = np.ones((M.shape[1], 64, 64))
HTones = HT(np.ones((M.shape[0], 64, 64)), M, sigmas)
iterations = np.ones((num_iters, M.shape[1], 64, 64))
for i in range(num_iters):
    Hest = H(est, M, sigmas)
    ratio = ims / (Hest + 1E-12)
    HTratio = HT(ratio, M, sigmas)
    est = est * HTratio / HTones
    iterations[i, :, :, :] = est
tifffile.imwrite('demo_unmixed_deconvolution.tif', est)
tifffile.imwrite('demo_unmixed_deconvolution_iterations.tif', iterations.astype('float32'), imagej=True)
