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

def numpy_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two arrays using NumPy."""
    return np.add(a, b)


def numpy_gaussian(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian filter using SciPy (NumPy backend)."""
    return ndi.gaussian_filter(data, sigma=sigma)


def numpy_threshold(data: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Apply threshold using NumPy."""
    return (data > threshold).astype(np.float32)


# ============================================================================
# CuPy Operations
# ============================================================================

def cupy_add(a, b):
    """Add two arrays using CuPy."""
    import cupy as cp
    result = cp.add(a, b)
    cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
    return result


def cupy_gaussian(data, sigma: float = 1.0):
    """Apply Gaussian filter using CuPy."""
    import cupyx.scipy.ndimage as cpndi
    result = cpndi.gaussian_filter(data, sigma=sigma)
    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    return result


def cupy_threshold(data, threshold: float = 0.5):
    """Apply threshold using CuPy."""
    import cupy as cp
    result = (data > threshold).astype(cp.float32)
    cp.cuda.Stream.null.synchronize()
    return result


# ============================================================================
# pyclesperanto Operations
# ============================================================================

def cle_add(a, b):
    """Add two arrays using pyclesperanto."""
    import pyclesperanto as cle
    return cle.add_images_weighted(a, b, factor1=1.0, factor2=1.0)


def cle_gaussian(data, sigma: float = 1.0):
    """Apply Gaussian filter using pyclesperanto."""
    import pyclesperanto as cle
    # Only apply sigma to x and y dimensions for 2D data
    if len(data.shape) == 2:
        return cle.gaussian_blur(data, sigma_x=sigma, sigma_y=sigma, sigma_z=0)
    else:
        return cle.gaussian_blur(data, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)


def cle_threshold(data, threshold: float = 0.5):
    """Apply threshold using pyclesperanto."""
    import pyclesperanto as cle
    return cle.greater_constant(data, constant=threshold)
