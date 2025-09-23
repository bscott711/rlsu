#!/usr/bin/env python

# Richardson-Lucy Deconvolution Spectral Unmixing (Portable & Refactored)
# Original by James Manton, 2022
# Refactoring by Gemini, 2025

import time
import tifffile
from pandas import read_csv
import numpy as np

# --- Backend Selection: Placed here for global access by helpers ---
try:
    import cupy as xp # type: ignore
    GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    import numpy as xp
    GPU_AVAILABLE = False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def asarray(data, dtype=xp.float32):
    """Move data to the selected backend (GPU or CPU)."""
    return xp.asarray(data, dtype=dtype)

def asnumpy(data):
    """Move data back to the CPU if it's on the GPU."""
    if GPU_AVAILABLE:
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

def _H_fft(x, M, psf_ft_array):
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

def _HT_fft(x, M, psf_ft_array):
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

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def deconv_unmix(
    mixed_image: np.ndarray,
    mixing_matrix: np.ndarray,
    sigmas: np.ndarray,
    num_iters: int = 100,
    return_iterations: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Performs joint deconvolution and spectral unmixing using the Richardson-Lucy algorithm.

    Args:
        mixed_image (np.ndarray): The input image data as a 4D NumPy array (Z, C, Y, X).
        mixing_matrix (np.ndarray): The spectral mixing matrix (C, O).
        sigmas (np.ndarray): An array of Gaussian sigma values for the blur in each channel.
        num_iters (int): The number of iterations to perform.
        return_iterations (bool): If True, returns the stack of all intermediate iterations.

    Returns:
        tuple[np.ndarray, np.ndarray | None]: A tuple containing:
            - The final unmixed and deconvolved image (Z, O, Y, X).
            - The stack of all iterations (I, Z, O, Y, X) if return_iterations is True, otherwise None.
    """
    if GPU_AVAILABLE:
        print("✅ CuPy backend detected. Running on NVIDIA GPU.")
    else:
        print("ℹ️ CuPy not found. Running on NumPy backend (CPU).")

    if mixed_image.ndim != 4:
        raise ValueError("Input image must be a 4D array (Z, C, Y, X).")
        
    num_z, num_c, num_y, num_x = mixed_image.shape
    SHAPE = (num_y, num_x)
    num_channels, num_objects = mixing_matrix.shape

    # --- Move data to the selected backend (GPU or CPU) ---
    M = asarray(mixing_matrix)
    ims = asarray(mixed_image)

    # --- Pre-computation Step ---
    print("Pre-computing FFT of PSFs...")
    psf_ft = xp.zeros((num_channels, SHAPE[0], SHAPE[1] // 2 + 1), dtype=xp.complex64)
    for c in range(num_channels):
        psf_ft[c] = create_gaussian_psf_fft(sigmas[c], SHAPE)

    # --- Richardson-Lucy Iteration ---
    print(f"Starting Richardson-Lucy iterations for {num_z} slice(s)...")
    start_time = time.time()
    
    # Process the entire Z-stack at once
    est = xp.ones((num_z, num_objects, num_y, num_x), dtype=xp.float32)
    ones = xp.ones((num_z, num_channels, num_y, num_x), dtype=xp.float32)
    HTones = _HT_fft(ones, M, psf_ft)

    iterations_cpu = None
    if return_iterations:
        iterations_cpu = np.zeros((num_iters, num_z, num_objects, num_y, num_x), dtype=np.float32)

    for i in range(num_iters):
        print(f"Iteration {i+1}/{num_iters}", end='\r')
        Hest = _H_fft(est, M, psf_ft)
        ratio = ims / (Hest + 1E-12)
        HTratio = _HT_fft(ratio, M, psf_ft)
        est = est * HTratio / (HTones + 1e-12)

        if return_iterations:
            iterations_cpu[i] = asnumpy(est)

    total_time = time.time() - start_time
    print(f"\nFinished {num_iters} iterations in {total_time:.2f} seconds.")

    return asnumpy(est), iterations_cpu


# =============================================================================
# EXAMPLE USAGE BLOCK
# =============================================================================

if __name__ == '__main__':
    print("--- Running Demo ---")
    
    # 1. Load demo data from files
    # Note: Using the original blurred/mixed data for a true test
    mixed_data_cpu = tifffile.imread('demo_raw_data_portable.tif')
    if mixed_data_cpu.ndim == 3: # Ensure it's a 4D stack
        mixed_data_cpu = mixed_data_cpu[np.newaxis, ...]
        
    M_cpu = read_csv('demo_PRISM_mixing_matrix_EGFP_Qdot_525_Qdot_565_Qdot_585_Qdot_605_Qdot_655_Qdot_705_TagBFP.csv', header=0).values
    sigmas_cpu = np.array([491, 514, 532, 561, 594, 633, 670, 710]) / 491

    # 2. Call the main processing function
    final_result, iterations_stack = deconv_unmix(
        mixed_image=mixed_data_cpu.astype(np.float32),
        mixing_matrix=M_cpu.astype(np.float32),
        sigmas=sigmas_cpu,
        num_iters=100,
        return_iterations=True,
    )
    
    # 3. Save the returned results to files
    print("\n--- Saving Demo Results ---")
    if final_result is not None:
        tifffile.imwrite('demo_final_result.tif', final_result)
        print("Final result saved to demo_final_result.tif")
        
    if iterations_stack is not None:
        tifffile.imwrite('demo_all_iterations.tif', iterations_stack, imagej=True)
        print("Iteration stack saved to demo_all_iterations.tif")