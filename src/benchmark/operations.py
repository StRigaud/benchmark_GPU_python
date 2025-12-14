"""
Benchmark operations for comparing numpy, cupy, and pyclesperanto.

This module provides wrapped functions for common operations that can be
benchmarked across different backends.
"""

import numpy as np
from scipy import ndimage as ndi


# ============================================================================
# NumPy Operations
# ============================================================================

def numpy_elementwise_simple(a: np.ndarray) -> np.ndarray:
    """Add two arrays using NumPy."""
    return a ** 2

def numpy_elementwise(a: np.ndarray) -> np.ndarray:
    """Add two arrays using NumPy."""
    return np.sin(a) ** 2 + np.cos(a) ** 2


def numpy_gaussian(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian filter using SciPy (NumPy backend)."""
    return ndi.gaussian_filter(data, sigma=sigma)


def numpy_slicing(data: np.ndarray) -> np.ndarray:
    """Apply threshold using NumPy."""
    return data[::3].copy()

def numpy_sum(data: np.ndarray) -> float:
    """Compute sum using NumPy."""
    return np.sum(data)


def numpy_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication using NumPy."""
    return np.matmul(a, b)


def numpy_std(data: np.ndarray) -> float:
    """Compute standard deviation using NumPy."""
    return np.std(data)


def numpy_fft(data: np.ndarray) -> np.ndarray:
    """Compute FFT using NumPy."""
    return np.fft.fftn(data)


def numpy_convolve(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply convolution using NumPy/SciPy."""
    return ndi.convolve(data, kernel)


# ============================================================================
# CuPy Operations
# ============================================================================

def cupy_elementwise_simple(a):
    """Add two arrays using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    result = a ** 2
    cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
    return result

def cupy_elementwise(a):
    """Add two arrays using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    result = cp.sin(a) ** 2 + cp.cos(a) ** 2
    cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
    return result


def cupy_gaussian(data, sigma: float = 1.0):
    """Apply Gaussian filter using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    import cupyx.scipy.ndimage as cpndi
    result = cpndi.gaussian_filter(data, sigma=sigma)
    cp.cuda.Stream.null.synchronize()
    return result


def cupy_slicing(data):
    """Apply threshold using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    result = data[::3].copy()
    cp.cuda.Stream.null.synchronize()
    return result

def cupy_sum(data):
    """Compute sum using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    result = cp.sum(data)
    cp.cuda.Stream.null.synchronize()
    return result


def cupy_matmul(a, b):
    """Matrix multiplication using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    result = cp.matmul(a, b)
    cp.cuda.Stream.null.synchronize()
    return result


def cupy_std(data):
    """Compute standard deviation using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    result = cp.std(data)
    cp.cuda.Stream.null.synchronize()
    return result


def cupy_fft(data):
    """Compute FFT using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    result = cp.fft.fftn(data)
    cp.cuda.Stream.null.synchronize()
    return result


def cupy_convolve(data, kernel):
    """Apply convolution using CuPy."""
    import cupy as cp
    cp.cuda.Device(-1)
    import cupyx.scipy.ndimage as cpndi
    result = cpndi.convolve(data, kernel)
    cp.cuda.Stream.null.synchronize()
    return result


# ============================================================================
# pyclesperanto Operations
# ============================================================================

def cle_elementwise_simple(a):
    """Add two arrays using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    b = cle.create_like(a)
    cle.power(a, scalar=2, output_image=b)
    return b

def cle_elementwise(a):
    """Add two arrays using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    b = cle.create_like(a)
    source = """
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void elementwise( IMAGE_src_TYPE  src, IMAGE_dst_TYPE  dst) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const float a = (float) READ_IMAGE(src, sampler, POS_src_INSTANCE(x, y, z, 0)).x;
    const float result = pow(sin(a), 2) + pow(cos(a), 2);
    WRITE_IMAGE(dst, POS_dst_INSTANCE(x, y, z, 0), CONVERT_dst_PIXEL_TYPE(result));
    }
    """
    params = { "src": a, "dst": b }
    cle.execute( kernel_source=source,
        kernel_name="elementwise",
        global_size=a.shape,
        parameters=params,
        local_size=(0,0,0))
    return b
    # return cle.power(cle.sin(a), scalar=2) + cle.power(cle.cos(a), scalar=2)


def cle_gaussian(data, sigma: float = 1.0):
    """Apply Gaussian filter using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    return cle.gaussian_blur(data, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)


def cle_slicing(data):
    """Apply threshold using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    return data[::3]


def cle_sum(data):
    """Compute sum using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    return cle.sum_of_all_pixels(data)


def cle_matmul(a, b):
    """Matrix multiplication using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    return cle.multiply_matrix(a, b)


def cle_std(data):
    """Compute standard deviation using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    return cle.standard_deviation_of_all_pixels(data)


def cle_fft(data):
    """Compute FFT using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    return cle.fft(data)


def cle_convolve(data, kernel):
    """Apply convolution using pyclesperanto."""
    import pyclesperanto as cle
    cle.select_device(1, "gpu")
    cle.wait_for_kernel_to_finish(True)
    return cle.convolve(data, kernel)
