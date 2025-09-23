#!/usr/bin/env python

# Richardson-Lucy Deconvolution Spectral Unmixing (Portable FFT-based)
# Original by James Manton, 2022
# FFT implementation and portability by Gemini, 2025

import time
import tifffile
from pandas import read_csv
import numpy as np # Import numpy at the top for allocation

# --- Backend Selection: Use CuPy if available, otherwise fall back to NumPy ---
try:
    import cupy as xp # type: ignore
    print("✅ CuPy backend detected. Running on NVIDIA GPU.")
except (ImportError, ModuleNotFoundError):
    import numpy as xp
    print("ℹ️ CuPy not found. Running on NumPy backend (CPU).")

# Helper functions for backend-agnostic data handling
def asarray(data, dtype=xp.float32):
    """Move data to the selected backend (GPU or CPU)."""
    return xp.asarray(data, dtype=dtype)

def asnumpy(data):
    """Move data back to the CPU if it's on the GPU."""
    if 'cupy' in xp.__name__:
        return data.get()
    return data

def create_gaussian_psf_fft(sigma, shape):
    """Creates the Fourier transform of a Gaussian PSF on the selected backend."""
    grid_y, grid_x = xp.mgrid[0:shape[0], 0:shape[1]]
    center_y, center_x = shape[0] // 2, shape[1] // 2
    psf = xp.exp(-((grid_y - center_y)**2 + (grid_x - center_x)**2) / (2 * sigma**2))
    psf = xp.fft.ifftshift(psf)
    psf_ft = xp.fft.rfft2(psf / psf.sum())
    return psf_ft

def H_fft(x, M, psf_ft_array):
    """Forward operator using FFT for convolution."""
    num_channels, num_objects = M.shape
    ims = xp.zeros((num_channels, x.shape[1], x.shape[2]), dtype=xp.float32)

    for o in range(num_objects):
        x_obj_ft = xp.fft.rfft2(x[o, :, :])
        for c in range(num_channels):
            blurred_ft = x_obj_ft * psf_ft_array[c]
            blurred = xp.fft.irfft2(blurred_ft, s=x.shape[1:])
            ims[c, :, :] += M[c, o] * blurred
    return ims

def HT_fft(x, M, psf_ft_array):
    """Adjoint operator using FFT for convolution."""
    num_channels, num_objects = M.shape
    ests = xp.zeros((num_objects, x.shape[1], x.shape[2]), dtype=xp.float32)

    for o in range(num_objects):
        obj_est = xp.zeros_like(ests[0])
        for c in range(num_channels):
            unmixed_channel = x[c, :, :] * M[c, o]
            unmixed_channel_ft = xp.fft.rfft2(unmixed_channel)
            back_projected_ft = unmixed_channel_ft * xp.conj(psf_ft_array[c])
            obj_est += xp.fft.irfft2(back_projected_ft, s=x.shape[1:])
        ests[o, :, :] = obj_est
    
    return ests / num_channels

# --- Main Script ---
def main():
    num_iters = 100
    SHAPE = (64, 64)

    # Load data from CPU using NumPy
    letters_cpu = tifffile.imread('demo_ground_truth.tif')
    M_cpu = read_csv('demo_PRISM_mixing_matrix_EGFP_Qdot_525_Qdot_565_Qdot_585_Qdot_605_Qdot_655_Qdot_705_TagBFP.csv', header=0).values
    sigmas_cpu = np.array([491, 514, 532, 561, 594, 633, 670, 710]) / 491

    # Move data to the selected backend (GPU or CPU)
    M = asarray(M_cpu)
    letters = asarray(letters_cpu)

    print("Pre-computing FFT of PSFs...")
    num_channels, num_objects = M.shape
    psf_ft = xp.zeros((num_channels, SHAPE[0], SHAPE[1] // 2 + 1), dtype=xp.complex64)
    for c in range(num_channels):
        psf_ft[c] = create_gaussian_psf_fft(sigmas_cpu[c], SHAPE)

    print("Generating blurred and mixed raw data...")
    ims = H_fft(letters, M, psf_ft)
    ims = xp.clip(ims, 0, None)
    if 'numpy' in xp.__name__:
        ims_cpu_noise = np.random.poisson(asnumpy(ims))
        ims = asarray(ims_cpu_noise)
    else:
        ims = xp.random.poisson(ims)
        
    tifffile.imwrite('demo_raw_data_portable.tif', asnumpy(ims))

    print("Starting Richardson-Lucy iterations...")
    start_time = time.time()
    
    est = xp.ones((num_objects, SHAPE[0], SHAPE[1]), dtype=xp.float32)
    ones = xp.ones((num_channels, SHAPE[0], SHAPE[1]), dtype=xp.float32)
    HTones = HT_fft(ones, M, psf_ft)

    iterations_cpu = np.zeros((num_iters, num_objects, SHAPE[0], SHAPE[1]), dtype=np.float32)

    for i in range(num_iters):
        print(f"Iteration {i+1}/{num_iters}", end='\r')
        Hest = H_fft(est, M, psf_ft)
        ratio = ims / (Hest + 1E-12)
        HTratio = HT_fft(ratio, M, psf_ft)
        est = est * HTratio / (HTones + 1e-12)

        iterations_cpu[i] = asnumpy(est)

    total_time = time.time() - start_time
    print(f"\nFinished {num_iters} iterations in {total_time:.2f} seconds.")

    # Save final and intermediate results
    print("Saving results...")
    tifffile.imwrite(
        'demo_iterations_portable.tif',
        iterations_cpu,
        imagej=True
    )

    est_cpu = asnumpy(est)
    tifffile.imwrite('demo_unmixed_deconvolution_portable.tif', est_cpu)
    print("All results saved.")

if __name__ == '__main__':
    main()
