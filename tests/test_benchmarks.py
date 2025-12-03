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
    numpy_matmul,
    numpy_std,
    numpy_fft,
    numpy_convolve,
    cupy_elementwise,
    cupy_gaussian,
    cupy_slicing,
    cupy_sum,
    cupy_matmul,
    cupy_std,
    cupy_fft,
    cupy_convolve,
    cle_elementwise,
    cle_gaussian,
    cle_slicing,
    cle_sum,
    cle_matmul,
    cle_std,
    cle_fft,
    cle_convolve,
)

# Check which backends are available
BACKENDS = check_backend_availability()

# Define test array sizes
SIZES = {
    "64MB": (256, 256, 256),
    "512MB": (512, 512, 512),
    # "4096MB": (1024, 1024, 1024),
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
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'numpy',
        'operation': 'elementwise'
    })
    numpy_elementwise(a) # Warm-up
    result = benchmark(numpy_elementwise, a)
    assert result.shape == a.shape


@skip_if_no_cupy
def test_elementwise_cupy(benchmark, cupy_arrays):
    """Benchmark cupy array elementwise operation."""
    a, _, size_name = cupy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'cupy',
        'operation': 'elementwise'
    })
    cupy_elementwise(a) # Warm-up
    result = benchmark(cupy_elementwise, a)
    assert result.shape == a.shape


@skip_if_no_cle
def test_elementwise_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto array elementwise operation."""
    a, _, size_name = cle_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'pyclesperanto',
        'operation': 'elementwise'
    })
    cle_elementwise(a) # Warm-up
    result = benchmark(cle_elementwise, a)
    assert result.shape == a.shape

# ============================================================================
# Gaussian Filter Benchmarks
# ============================================================================

def test_gaussian_numpy(benchmark, numpy_arrays):
    """Benchmark numpy gaussian filter."""
    a, _, size_name = numpy_arrays
    s = 15.0
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'numpy',
        'operation': 'gaussian({sigma})'.format(sigma=s)
    })
    numpy_gaussian(a, sigma=s) # Warm-up
    result = benchmark(numpy_gaussian, a, sigma=s)
    assert result.shape == a.shape


@skip_if_no_cupy
def test_gaussian_cupy(benchmark, cupy_arrays):
    """Benchmark cupy gaussian filter."""
    a, _, size_name = cupy_arrays
    s = 15.0
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'cupy',
        'operation': 'gaussian({sigma})'.format(sigma=s)
    })
    cupy_gaussian(a, sigma=s) # Warm-up
    result = benchmark(cupy_gaussian, a, sigma=s)
    assert result.shape == a.shape


@skip_if_no_cle
def test_gaussian_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto gaussian filter."""
    a, _, size_name = cle_arrays
    s = 15.0
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'pyclesperanto',
        'operation': 'gaussian({sigma})'.format(sigma=s)
    })
    cle_gaussian(a, sigma=s) # Warm-up
    result = benchmark(cle_gaussian, a, sigma=s)
    assert result.shape == a.shape


# ============================================================================
# Slicing Benchmarks
# ============================================================================

def test_slicing_numpy(benchmark, numpy_arrays):
    """Benchmark numpy slicing."""
    a, _, size_name = numpy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'numpy',
        'operation': 'slicing'
    })
    numpy_slicing(a) # Warm-up
    result = benchmark(numpy_slicing, a)
    assert True


@skip_if_no_cupy
def test_slicing_cupy(benchmark, cupy_arrays):
    """Benchmark cupy slicing."""
    a, _, size_name = cupy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'cupy',
        'operation': 'slicing'
    })
    cupy_slicing(a) # Warm-up
    result = benchmark(cupy_slicing, a)
    assert True


@skip_if_no_cle
def test_slicing_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto slicing."""
    a, _, size_name = cle_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'pyclesperanto',
        'operation': 'slicing'
    })
    cle_slicing(a) # Warm-up
    result = benchmark(cle_slicing, a)
    assert True

# ============================================================================
# Sum Benchmarks
# ============================================================================

def test_sum_numpy(benchmark, numpy_arrays):
    """Benchmark numpy array sum operation."""
    a, _, size_name = numpy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'numpy',
        'operation': 'sum'
    })
    numpy_sum(a) # Warm-up
    result = benchmark(numpy_sum, a)
    assert np.isscalar(result)


@skip_if_no_cupy
def test_sum_cupy(benchmark, cupy_arrays):
    """Benchmark cupy array sum operation."""
    a, _, size_name = cupy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'cupy',
        'operation': 'sum'
    })
    cupy_sum(a) # Warm-up
    result = benchmark(cupy_sum, a)
    assert np.isscalar(result.get())


@skip_if_no_cle
def test_sum_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto array sum operation."""
    a, _, size_name = cle_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'pyclesperanto',
        'operation': 'sum'
    })
    cle_sum(a) # Warm-up
    result = benchmark(cle_sum, a)
    assert np.isscalar(result)

# ============================================================================
# Matmul Benchmarks
# ============================================================================

def test_matmul_numpy(benchmark, numpy_arrays):
    """Benchmark numpy matrix multiplication."""
    a, b, size_name = numpy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'numpy',
        'operation': 'matmul'
    })
    numpy_matmul(a, b) # Warm-up
    result = benchmark(numpy_matmul, a, b)
    assert result.shape == a.shape 

@skip_if_no_cupy
def test_matmul_cupy(benchmark, cupy_arrays):
    """Benchmark cupy matrix multiplication."""
    a, b, size_name = cupy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'cupy',
        'operation': 'matmul'
    })
    cupy_matmul(a, b) # Warm-up
    result = benchmark(cupy_matmul, a, b)
    assert result.shape == a.shape

@skip_if_no_cle
def test_matmul_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto matrix multiplication."""
    a, b, size_name = cle_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'pyclesperanto',
        'operation': 'matmul'
    })
    cle_matmul(a, b) # Warm-up
    result = benchmark(cle_matmul, a, b)
    assert result.shape == a.shape

# ============================================================================
# Standard Deviation Benchmarks
# ============================================================================

def test_std_numpy(benchmark, numpy_arrays):
    """Benchmark numpy array standard deviation operation."""
    a, _, size_name = numpy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'numpy',
        'operation': 'std'
    })
    numpy_std(a) # Warm-up
    result = benchmark(numpy_std, a)
    assert np.isscalar(result)


@skip_if_no_cupy
def test_std_cupy(benchmark, cupy_arrays):
    """Benchmark cupy array standard deviation operation."""
    a, _, size_name = cupy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'cupy',
        'operation': 'std'
    })
    cupy_std(a) # Warm-up
    result = benchmark(cupy_std, a)
    assert np.isscalar(result.get())


@skip_if_no_cle
def test_std_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto array standard deviation operation."""
    a, _, size_name = cle_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'pyclesperanto',
        'operation': 'std'
    })
    cle_std(a) # Warm-up
    result = benchmark(cle_std, a)
    assert np.isscalar(result)

# ============================================================================
# FFT Benchmarks
# ============================================================================

def test_fft_numpy(benchmark, numpy_arrays):
    """Benchmark numpy FFT operation."""
    a, _, size_name = numpy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'numpy',
        'operation': 'fft'
    })
    numpy_fft(a) # Warm-up
    result = benchmark(numpy_fft, a)
    assert True


@skip_if_no_cupy
def test_fft_cupy(benchmark, cupy_arrays):
    """Benchmark cupy FFT operation."""
    a, _, size_name = cupy_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'cupy',
        'operation': 'fft'
    })
    cupy_fft(a) # Warm-up
    result = benchmark(cupy_fft, a)
    assert True


@skip_if_no_cle
def test_fft_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto FFT operation."""
    a, _, size_name = cle_arrays
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'pyclesperanto',
        'operation': 'fft'
    })
    cle_fft(a) # Warm-up
    result = benchmark(cle_fft, a)
    assert True

# ============================================================================
# Convolution Benchmarks
# ============================================================================

def test_convolve_numpy(benchmark, numpy_arrays):
    """Benchmark numpy convolution operation."""
    a, _, size_name = numpy_arrays
    # Create a nxnxn kernel
    n = 7
    kernel = np.ones((n, n, n), dtype=np.float32) / (n ** 3)
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'numpy',
        'operation': 'convolve({n})'.format(n=n)
    })
    numpy_convolve(a, kernel) # Warm-up
    result = benchmark(numpy_convolve, a, kernel)
    assert result.shape == a.shape


@skip_if_no_cupy
def test_convolve_cupy(benchmark, cupy_arrays):
    """Benchmark cupy convolution operation."""
    import cupy as cp
    a, _, size_name = cupy_arrays
    # Create a nxnxn kernel
    n = 7
    kernel = cp.ones((n, n, n), dtype=cp.float32) / (n ** 3)
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'cupy',
        'operation': 'convolve({n})'.format(n=n)
    })
    cupy_convolve(a, kernel) # Warm-up
    result = benchmark(cupy_convolve, a, kernel)
    assert result.shape == a.shape


@skip_if_no_cle
def test_convolve_pyclesperanto(benchmark, cle_arrays):
    """Benchmark pyclesperanto convolution operation."""
    import pyclesperanto as cle
    a, _, size_name = cle_arrays
    # Create a nxnxn kernel
    n = 7
    kernel_np = np.ones((n, n, n), dtype=np.float32) / (n ** 3)
    kernel = cle.push(kernel_np)
    benchmark.extra_info.update({
        'size': size_name,
        'size_shape': SIZES[size_name],
        'backend': 'pyclesperanto',
        'operation': 'convolve({n})'.format(n=n)
    })
    cle_convolve(a, kernel) # Warm-up
    result = benchmark(cle_convolve, a, kernel)
    assert result.shape == a.shape