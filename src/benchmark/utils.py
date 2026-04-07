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
        "pyclesperanto_cuda": False,
        "pyclesperanto_metal": False,
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
        # cupy.cuda.Device(0).compute_capability
        backends["cupy"] = True
    except (ImportError, Exception):
        pass
    
    # Check pyclesperanto
    try:
        import pyclesperanto as cle
        backends["pyclesperanto"] = True
    except ImportError:
        pass
    
    # Check pyclesperanto cuda
    try:
        import pyclesperanto as cle
        if "cuda" in cle.list_available_backends():
            backends["pyclesperanto_cuda"] = True
    except ImportError:
        pass

    # Check pyclesperanto metal
    try:
        import pyclesperanto as cle
        if "metal" in cle.list_available_backends():
            backends["pyclesperanto_metal"] = True
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
        cp.cuda.Device(-1)
        return cp.asarray(data)
    elif backend == "pyclesperanto":
        import pyclesperanto as cle
        cle.select_backend("opencl")
        cle.select_device(1, "gpu")
        cle.wait_for_kernel_to_finish(True)
        return cle.push(data)
    elif backend == "pyclesperanto_cuda":
        import pyclesperanto as cle
        cle.select_backend("cuda")
        cle.select_device(1, "gpu")
        cle.wait_for_kernel_to_finish(True)
        return cle.push(data)    
    elif backend == "pyclesperanto_metal":
        import pyclesperanto as cle
        cle.select_backend("metal")
        cle.select_device(1, "gpu")
        cle.wait_for_kernel_to_finish(True)
        return cle.push(data)
    else:
        raise ValueError(f"Unknown backend: {backend}")
