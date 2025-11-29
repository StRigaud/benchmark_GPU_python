# benchmark_GPU_python

A simple, non‑exhaustive toolkit for benchmarking basic operations on GPU vs CPU. It compares GPU libraries (pyclesperanto, CuPy) with CPU libraries (NumPy, SciPy).

## Installation

### Prerequisites

Install [pixi](https://pixi.sh) first:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/StRigaud/benchmark_GPU_python.git
cd benchmark_GPU_python
pixi install
pixi run install
```

## Usage

```bash
pixi run benchmark  # Running benchmarking
pixi run charts    # Generate graphics from benchmarks
```

## Benchmark Operations

The library benchmarks the following operations:

| Operation | Description |
|-----------|-------------|
| `elementwise` | Scalar operation on all elements of the array |
| `gaussian` | Convolution operation with a Gaussian kernel (𝜎=3) |
| `slicing` | Select every third element of first dimension |
| `sum` | Compute sum of entire array, reducing it to a single scalar |
| `std` | Compute std of entire array, reducing it to a single scalar |
| `matmul` | Multiplication of two square matrices |
| `fft` | Fast Fourier Transform of matrix |

## Project Structure

```
benchmark_GPU_python/
├── pixi.toml           # Pixi project configuration
├── pyproject.toml      # Python package configuration
├── src/
│   └── benchmark/
│       ├── __init__.py
│       ├── operations.py   # Benchmark operations for each backend
│       ├── utils.py        # Utility functions
│       └── charts.py       # Chart generation
└── tests/
    └── test_benchmarks.py  # pytest-benchmark tests
```

## Development

### Adding New Operations

1. Add the operation functions to `src/benchmark/operations.py` for each backend
2. Add corresponding benchmark tests in `tests/test_benchmarks.py`
3. Run the benchmarks to verify

### Incompatibilities

This code used `cupy` which is restricted to NVIDIA hardware. 

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.