"""
Benchmark tests comparing numpy, cupy, and pyclesperanto performance.

Run with: pixi run test
Or for benchmark-only: pixi run benchmark
"""

import numpy as np
import pytest

from benchmark.utils import check_backend_availability, generate_test_data
from benchmark.operations import (
    numpy_elementwise,
    numpy_gaussian,
    numpy_slicing,
    numpy_sum,
    cupy_elementwise,
    cupy_gaussian,
    cupy_slicing,
    cupy_sum,
    cle_elementwise,
    cle_gaussian,
    cle_slicing,
    cle_sum,
)

# Check which backends are available
BACKENDS = check_backend_availability()

# Define test array sizes
SIZES = {
    "small": (256, 256, 256),
    "medium": (512, 512, 512),
    # "large": (1024, 1024, 1024),
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
# Elementwise Benchmarks
# ============================================================================

def test_elementwise_numpy(benchmark, numpy_arrays):
    """Benchmark numpy array elementwise operation."""
    a, _, size_name = numpy_arrays
    result = benchmark(numpy_elementwise, a)
    assert result.shape == a.shape


@skip_if_no_cupy
def test_elementwise_cupy(benchmark, cupy_arrays):
    """Benchmark cupy array elementwise operation."""
    a, _, size_name = cupy_arrays
    result = benchmark(cupy_elementwise, a)
    assert result.shape == a.shape


@skip_if_no_cle
def test_elementwise_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto array elementwise operation."""
    a, _, size_name = cle_arrays
    result = benchmark(cle_elementwise, a)
    assert result.shape == a.shape

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
# Slicing Benchmarks
# ============================================================================

def test_slicing_numpy(benchmark, numpy_arrays):
    """Benchmark numpy slicing."""
    a, _, size_name = numpy_arrays
    result = benchmark(numpy_slicing, a)
    assert True


@skip_if_no_cupy
def test_slicing_cupy(benchmark, cupy_arrays):
    """Benchmark cupy slicing."""
    a, _, size_name = cupy_arrays
    result = benchmark(cupy_slicing, a)
    assert True


@skip_if_no_cle
def test_slicing_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto slicing."""
    a, _, size_name = cle_arrays
    result = benchmark(cle_slicing, a)
    assert True

# ============================================================================
# Sum Benchmarks
# ============================================================================

def test_sum_numpy(benchmark, numpy_arrays):
    """Benchmark numpy array sum operation."""
    a, _, size_name = numpy_arrays
    result = benchmark(numpy_sum, a)
    assert np.isscalar(result)


@skip_if_no_cupy
def test_sum_cupy(benchmark, cupy_arrays):
    """Benchmark cupy array sum operation."""
    a, _, size_name = cupy_arrays
    result = benchmark(cupy_sum, a)
    assert np.isscalar(result)


@skip_if_no_cle
def test_sum_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto array sum operation."""
    a, _, size_name = cle_arrays
    result = benchmark(cle_sum, a)
    assert np.isscalar(result)