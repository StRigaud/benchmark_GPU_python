"""
Benchmark tests comparing numpy, cupy, and pyclesperanto performance.

Run with: pixi run test
Or for benchmark-only: pixi run benchmark
"""

import numpy as np
import pytest

from benchmark import operations as ops
from benchmark.utils import (
    BACKEND_LABELS,
    check_backend_availability,
    generate_test_data,
    get_device_info,
    to_backend,
)

BACKENDS = check_backend_availability()

SIZES = {
    "128MB": (128, 512, 512),
    "2048MB": (128, 2048, 2048),
}

SIGMA = 7.0
KERNEL_N = 7

# operation label (stored in the JSON) -> function name suffix in operations.py
OPERATIONS = {
    "elementwise_simple": "elementwise_simple",
    "elementwise": "elementwise",
    f"gaussian({SIGMA})": "gaussian",
    "slicing (stridded)": "slicing",
    "sum": "sum",
    "matmul (2d)": "matmul",
    "std": "std",
    "fft": "fft",
    f"convolve({KERNEL_N})": "convolve",
}


def get_function(backend: str, func_name: str):
    """Look up numpy_*, cupy_* or cle_* in operations.py."""
    prefix = {"numpy": "numpy", "cupy": "cupy"}.get(backend, "cle")
    return getattr(ops, f"{prefix}_{func_name}")


@pytest.mark.parametrize("size_name", list(SIZES))
@pytest.mark.parametrize("operation", list(OPERATIONS))
@pytest.mark.parametrize("backend", list(BACKEND_LABELS))
def test_benchmark(benchmark, backend, operation, size_name):
    """Benchmark one operation on one backend and one array size."""
    if not BACKENDS[backend]:
        pytest.skip(f"{backend} not available")

    shape = SIZES[size_name]
    func_name = OPERATIONS[operation]
    func = get_function(backend, func_name)

    # Build the inputs (only allocate what the operation needs)
    if func_name == "matmul":
        a = generate_test_data(shape[1:], backend)  # 2D matrices
        b = generate_test_data(shape[1:], backend)
        args, kwargs = (a, b), {}
    elif func_name == "convolve":
        a = generate_test_data(shape, backend)
        kernel = np.ones((KERNEL_N,) * len(shape), dtype=np.float32) / KERNEL_N ** len(shape)
        args, kwargs = (a, to_backend(kernel, backend)), {}
    elif func_name == "gaussian":
        a = generate_test_data(shape, backend)
        args, kwargs = (a,), {"sigma": SIGMA}
    else:
        a = generate_test_data(shape, backend)
        args, kwargs = (a,), {}

    benchmark.extra_info.update({
        "size": size_name,
        "size_shape": shape,
        "backend": BACKEND_LABELS[backend],
        "operation": operation,
    })
    benchmark.extra_info.update(get_device_info(backend))

    if backend != "numpy":
        func(*args, **kwargs)  # warm-up (GPU only: kernel compilation, allocations)
    result = benchmark(func, *args, **kwargs)

    # Basic sanity check on the output shape
    if func_name in ("elementwise_simple", "elementwise", "gaussian", "convolve", "matmul"):
        assert result.shape == a.shape
    elif func_name == "slicing":
        assert result.shape[0] == len(range(0, shape[0], 3))
