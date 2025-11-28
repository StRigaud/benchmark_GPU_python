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


# ============================================================================
# CuPy Operations
# ============================================================================

def cupy_elementwise(a):
    """Add two arrays using CuPy."""
    import cupy as cp
    result = cp.sin(a) ** 2 + cp.cos(a) ** 2
    cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
    return result


def cupy_gaussian(data, sigma: float = 1.0):
    """Apply Gaussian filter using CuPy."""
    import cupy as cp
    import cupyx.scipy.ndimage as cpndi
    result = cpndi.gaussian_filter(data, sigma=sigma)
    cp.cuda.Stream.null.synchronize()
    return result


def cupy_slicing(data):
    """Apply threshold using CuPy."""
    import cupy as cp
    result = data[::3].copy()
    cp.cuda.Stream.null.synchronize()
    return result

def cupy_sum(data):
    """Compute sum using CuPy."""
    import cupy as cp
    result = cp.sum(data)
    cp.cuda.Stream.null.synchronize()
    return result


def cupy_matmul(a, b):
    """Matrix multiplication using CuPy."""
    import cupy as cp
    result = cp.matmul(a, b)
    cp.cuda.Stream.null.synchronize()
    return result


# ============================================================================
# pyclesperanto Operations
# ============================================================================

def cle_elementwise(a):
    """Add two arrays using pyclesperanto."""
    import pyclesperanto as cle
    cle.set_wait_for_kernel_finish(True)
    return cle.sin(a) ** 2 + cle.cos(a) ** 2


def cle_gaussian(data, sigma: float = 1.0):
    """Apply Gaussian filter using pyclesperanto."""
    import pyclesperanto as cle
    cle.set_wait_for_kernel_finish(True)
    return cle.gaussian_blur(data, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)


def cle_slicing(data):
    """Apply threshold using pyclesperanto."""
    import pyclesperanto as cle
    cle.set_wait_for_kernel_finish(True)
    return data[::3]

def cle_sum(data):
    """Compute sum using pyclesperanto."""
    import pyclesperanto as cle
    cle.set_wait_for_kernel_finish(True)
    return cle.sum_of_all_pixels(data)


def cle_matmul(a, b):
    """Matrix multiplication using pyclesperanto."""
    import pyclesperanto as cle
    cle.set_wait_for_kernel_finish(True)
    return cle.multiply_matrix(a, b)
