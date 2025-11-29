"""
Benchmark - A minimalist Python library for benchmarking GPU operations.

This library compares the performance of numpy, cupy, and pyclesperanto
for common image processing operations.
"""

from benchmark.operations import (
    numpy_elementwise,
    numpy_gaussian,
    numpy_slicing,
    numpy_sum,
    numpy_matmul,
    numpy_std,
    numpy_fft,
)
from benchmark.utils import check_backend_availability

__version__ = "0.1.0"

__all__ = [
    "numpy_elementwise",
    "numpy_gaussian",
    "numpy_slicing",
    "numpy_sum",
    "numpy_matmul",
    "numpy_std",
    "numpy_fft",
    "check_backend_availability",
]
