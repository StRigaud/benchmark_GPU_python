"""Utility functions for the benchmark library."""

import importlib

# Backend keys (used in test ids / -k filters) -> label stored in the benchmark JSON.
# The labels are what charts.py and boxplots.py expect.
BACKEND_LABELS = {
    "numpy": "numpy",
    "cupy": "cupy",
    "pyclesperanto": "pyclesperanto",  # OpenCL
    "pyclesperanto_cuda": "pyclesperanto (cuda)",
    "pyclesperanto_metal": "pyclesperanto (metal)",
}

DEVICE_TYPES = {
    "numpy": "CPU",
    "cupy": "GPU (NVIDIA)",
    "pyclesperanto": "GPU (OpenCL)",
    "pyclesperanto_cuda": "GPU (CUDA)",
    "pyclesperanto_metal": "GPU (Metal)",
}


CLE_BACKEND_NAMES = {
    "pyclesperanto": "opencl",
    "pyclesperanto_cuda": "cuda",
    "pyclesperanto_metal": "metal",
}


def check_backend_availability() -> dict[str, bool]:
    """Return {backend key: available} for every key in BACKEND_LABELS."""
    backends = {name: False for name in BACKEND_LABELS}
    backends["numpy"] = True

    try:
        import cupy
        backends["cupy"] = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        pass

    try:
        import pyclesperanto as cle
        available = cle.list_available_backends()
        backends["pyclesperanto"] = "opencl" in available
        backends["pyclesperanto_cuda"] = "cuda" in available
        backends["pyclesperanto_metal"] = "metal" in available
    except Exception:
        pass

    return backends


def to_backend(data, backend: str):
    """Move a numpy array to the given backend (no-op for numpy)."""
    if backend == "numpy":
        return data
    if backend == "cupy":
        import cupy as cp
        return cp.asarray(data)
    if backend.startswith("pyclesperanto"):
        import pyclesperanto as cle
        cle.select_backend(CLE_BACKEND_NAMES[backend])
        cle.select_device(0, "gpu")
        cle.wait_for_kernel_to_finish(True)
        return cle.push(data)
    raise ValueError(f"Unknown backend: {backend}")


def generate_test_data(size: tuple, backend: str = "numpy"):
    """Random float32 array of the given shape on the given backend."""
    import numpy as np

    rng = np.random.default_rng(42)
    return to_backend(rng.random(size, dtype=np.float32), backend)


def get_device_info(backend: str) -> dict[str, str]:
    """Return {'device_name', 'device_type', 'library_version'} for a backend key."""
    info = {"device_type": DEVICE_TYPES.get(backend, "Unknown")}
    lib = "numpy" if backend == "numpy" else "cupy" if backend == "cupy" else "pyclesperanto"
    try:
        info["library_version"] = f"{lib} {importlib.import_module(lib).__version__}"
    except Exception:
        info["library_version"] = f"{lib} unknown"
    try:
        if backend == "numpy":
            info["device_name"] = "CPU"
        elif backend == "cupy":
            import cupy as cp
            info["device_name"] = str(cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]).strip()
        else:
            import pyclesperanto as cle
            info["device_name"] = cle.get_device().name
    except Exception as e:
        info["device_name"] = f"Unknown ({type(e).__name__})"
    return info
