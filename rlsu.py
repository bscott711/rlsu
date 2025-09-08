#!/usr/bin/env python

# Richardson-Lucy Spectral Unmixing
# James Manton, 2022
# jmanton@mrc-lmb.cam.ac.uk

import numpy as np
import cupy as cp
import pandas as pd
import timeit
import tifffile
import argparse
import os


# Get input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--matrix', type = str, required = True)
parser.add_argument('--output', type = str, required = True)
parser.add_argument('--num_iters', type = int, default = 1000)
parser.add_argument('--inverse', type = str, required = False)
args = parser.parse_args()

# Read mixing matrix and calculate transpose
M = pd.read_csv(args.matrix, header=0)
M = cp.array(M, dtype=cp.float32, order='F')
MT = cp.array(cp.transpose(M), dtype=cp.float32)
if args.inverse is not None:
	Mpinv = np.linalg.pinv(M)
	Mpinv = Mpinv.get()

# Read mixed data
mixed = tifffile.imread(args.input)
mixed = mixed.astype('float32')

# Log which files we're using and what dimensionality we're working with
print('Input file: %s' % args.input)
print('Input shape: %s' % (mixed.shape, ))
print('Unmixing matrix file: %s' % args.matrix)
print('Matrix shape: %s' % (M.shape, ))
print('Output file: %s' % args.output)
print('Number of iterations: %d' % args.num_iters)
print('')

# Add new z-axis if we have 3D (i.e. one z-slice) data
if mixed.ndim == 3:
	mixed = np.expand_dims(mixed, axis=0)

# Reshape mixed data into vector
num_z = mixed.shape[0]
num_c = mixed.shape[1]
num_x = mixed.shape[2]
num_y = mixed.shape[3]
mixed = mixed.reshape(num_z, num_c, num_x * num_y)

# Precompute HTones
HTones = cp.matmul(MT, cp.ones_like(mixed[0]),)

# Calculate Richardson-Lucy iterations
recon = np.ones((num_z, M.shape[1], num_x * num_y))
if args.inverse is not None:
	inverse = np.ones_like(recon)
start_time = timeit.default_timer()
for z in range(num_z):
	slice = cp.array(mixed[z])
	recon_slice = cp.ones((M.shape[1], slice.shape[1]), dtype=cp.float32)

	if args.inverse is not None:
		inverse[z] = np.matmul(Mpinv, mixed[z])

	iter_start_time = timeit.default_timer()
	for iter in range(args.num_iters):
		Hu = cp.matmul(M, recon_slice)
		ratio = slice / (Hu + 1E-12)
		HTratio = cp.matmul(MT, ratio)
		recon_slice = recon_slice * HTratio / HTones
	
	recon[z] = recon_slice.get()
	calc_time = timeit.default_timer() - iter_start_time
	print("Slice %03d: %d iterations completed in %f s." % (z, args.num_iters, calc_time))

calc_time = timeit.default_timer() - start_time
print("Finished in %f s" % calc_time)

# Reshape and save unmixed data
recon = recon.reshape(num_z, M.shape[1], num_x, num_y)
tifffile.imwrite(args.output, recon, bigtiff=True)
print("\nUnmixed data written to %s" % args.output)

if args.inverse is not None:
	inverse = inverse.reshape(num_z, M.shape[1], num_x, num_y)
	tifffile.imwrite(args.inverse, inverse, bigtiff=True, metadata={'axes': 'ZCYX'})
	print("Linearly unmixed data written to %s" % args.inverse)
