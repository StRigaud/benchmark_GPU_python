"""Utility functions for the benchmark library."""

from typing import Dict


def check_backend_availability() -> Dict[str, bool]:
    """
    Check which backends are available.
    
    Returns:
        Dict with backend names as keys and availability as boolean values.
    """
    backends = {
        "numpy": False,
        "cupy": False,
        "pyclesperanto": False,
    }
    
    # NumPy should always be available
    try:
        import numpy
        backends["numpy"] = True
    except ImportError:
        pass
    
    # Check CuPy
    try:
        import cupy
        # Try to get a device to verify GPU is available
        cupy.cuda.Device(0).compute_capability
        backends["cupy"] = True
    except (ImportError, Exception):
        pass
    
    # Check pyclesperanto
    try:
        import pyclesperanto as cle
        backends["pyclesperanto"] = True
    except ImportError:
        pass
    
    return backends


def generate_test_data(size: tuple, backend: str = "numpy"):
    """
    Generate test data for benchmarking.
    
    Args:
        size: Tuple specifying the shape of the array.
        backend: The backend to use ("numpy", "cupy", or "pyclesperanto").
    
    Returns:
        Array of the specified size and backend type.
    """
    import numpy as np
    
    rng = np.random.default_rng(42)
    data = rng.random(size, dtype=np.float32)
    
    if backend == "numpy":
        return data
    elif backend == "cupy":
        import cupy as cp
        return cp.asarray(data)
    elif backend == "pyclesperanto":
        import pyclesperanto as cle
        return cle.push(data)
    else:
        raise ValueError(f"Unknown backend: {backend}")
