"""
Benchmark tests comparing numpy, cupy, and pyclesperanto performance.

Run with: pixi run test
Or for benchmark-only: pixi run benchmark
"""

import numpy as np
import pytest

from benchmark.utils import check_backend_availability, generate_test_data
from benchmark.operations import (
    numpy_add,
    numpy_gaussian,
    numpy_threshold,
    cupy_add,
    cupy_gaussian,
    cupy_threshold,
    cle_add,
    cle_gaussian,
    cle_threshold,
)

# Check which backends are available
BACKENDS = check_backend_availability()

# Define test array sizes
SIZES = {
    "small": (256, 256),
    "medium": (512, 512),
    "large": (1024, 1024),
}


# Skip decorators for unavailable backends
skip_if_no_cupy = pytest.mark.skipif(
    not BACKENDS["cupy"],
    reason="CuPy not available"
)

skip_if_no_cle = pytest.mark.skipif(
    not BACKENDS["pyclesperanto"],
    reason="pyclesperanto not available"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=list(SIZES.keys()))
def size_name(request):
    """Parameterized fixture for array sizes."""
    return request.param


@pytest.fixture
def numpy_arrays(size_name):
    """Generate numpy test arrays."""
    size = SIZES[size_name]
    a = generate_test_data(size, "numpy")
    b = generate_test_data(size, "numpy")
    return a, b, size_name


@pytest.fixture
def cupy_arrays(size_name):
    """Generate cupy test arrays."""
    if not BACKENDS["cupy"]:
        pytest.skip("CuPy not available")
    size = SIZES[size_name]
    a = generate_test_data(size, "cupy")
    b = generate_test_data(size, "cupy")
    return a, b, size_name


@pytest.fixture
def cle_arrays(size_name):
    """Generate pyclesperanto test arrays."""
    if not BACKENDS["pyclesperanto"]:
        pytest.skip("pyclesperanto not available")
    size = SIZES[size_name]
    a = generate_test_data(size, "pyclesperanto")
    b = generate_test_data(size, "pyclesperanto")
    return a, b, size_name


# ============================================================================
# Addition Benchmarks
# ============================================================================

def test_add_numpy(benchmark, numpy_arrays):
    """Benchmark numpy array addition."""
    a, b, size_name = numpy_arrays
    result = benchmark(numpy_add, a, b)
    assert result.shape == a.shape


@skip_if_no_cupy
def test_add_cupy(benchmark, cupy_arrays):
    """Benchmark cupy array addition."""
    a, b, size_name = cupy_arrays
    result = benchmark(cupy_add, a, b)
    assert result.shape == a.shape


@skip_if_no_cle
def test_add_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto array addition."""
    a, b, size_name = cle_arrays
    result = benchmark(cle_add, a, b)


# ============================================================================
# Gaussian Filter Benchmarks
# ============================================================================

def test_gaussian_numpy(benchmark, numpy_arrays):
    """Benchmark numpy gaussian filter."""
    a, _, size_name = numpy_arrays
    result = benchmark(numpy_gaussian, a, sigma=2.0)
    assert result.shape == a.shape


@skip_if_no_cupy
def test_gaussian_cupy(benchmark, cupy_arrays):
    """Benchmark cupy gaussian filter."""
    a, _, size_name = cupy_arrays
    result = benchmark(cupy_gaussian, a, sigma=2.0)
    assert result.shape == a.shape


@skip_if_no_cle
def test_gaussian_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto gaussian filter."""
    a, _, size_name = cle_arrays
    result = benchmark(cle_gaussian, a, sigma=2.0)


# ============================================================================
# Threshold Benchmarks
# ============================================================================

def test_threshold_numpy(benchmark, numpy_arrays):
    """Benchmark numpy threshold."""
    a, _, size_name = numpy_arrays
    result = benchmark(numpy_threshold, a, threshold=0.5)
    assert result.shape == a.shape


@skip_if_no_cupy
def test_threshold_cupy(benchmark, cupy_arrays):
    """Benchmark cupy threshold."""
    a, _, size_name = cupy_arrays
    result = benchmark(cupy_threshold, a, threshold=0.5)
    assert result.shape == a.shape


@skip_if_no_cle
def test_threshold_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto threshold."""
    a, _, size_name = cle_arrays
    result = benchmark(cle_threshold, a, threshold=0.5)
