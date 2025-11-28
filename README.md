# benchmark_GPU_python

A minimalist Python library for benchmarking GPU operations comparing numpy, cupy, and pyclesperanto.

## Features

- **pytest-based benchmarks**: Uses pytest-benchmark for reliable performance measurements
- **Multiple backends**: Compare numpy (CPU), cupy (CUDA GPU), and pyclesperanto (OpenCL GPU)
- **Chart generation**: Automatically generate comparison charts from benchmark results
- **pixi-based**: Easy installation and environment management with pixi

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
```

For GPU support (cupy and pyclesperanto):

```bash
pixi install -e gpu
```

## Usage

### Running Benchmarks

Run all benchmarks:

```bash
pixi run test
```

Run benchmarks with detailed output:

```bash
pixi run benchmark
```

Save benchmark results to JSON for chart generation:

```bash
pixi run pytest tests/test_benchmarks.py --benchmark-json=results.json
```

### Generating Charts

After running benchmarks with JSON output:

```bash
pixi run charts --input results.json --output ./charts
```

Or programmatically:

```python
from benchmark.charts import load_benchmark_results, create_comparison_chart

df = load_benchmark_results("results.json")
create_comparison_chart(df, output_path="comparison.png")
```

## Benchmark Operations

The library benchmarks the following operations:

| Operation | Description |
|-----------|-------------|
| `add` | Element-wise array addition |
| `gaussian` | Gaussian blur filter |
| `threshold` | Binary thresholding |

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

### Running Tests

```bash
pixi run test
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.