"""
Publication-style boxplot generation for benchmark results gathered across
multiple systems.

Measurements from every input JSON file (one per benchmarked system) are
pooled together per backend/operation — the plot does not break results down
per system, only per operation (x-axis) and backend (color-coded, dodged
boxes) — which keeps axis labels short and readable regardless of how many
systems contributed data.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

sns.set_theme(style="whitegrid", context="talk")

BACKEND_SHORT_NAMES = {
    "cupy": "CuPy",
    "numpy": "NumPy",
    "pyclesperanto": "pycle (OpenCL)",
    "pyclesperanto (cuda)": "pycle (CUDA)",
    "pyclesperanto (metal)": "pycle (Metal)",
}

BACKEND_ORDER = [
    "numpy",
    "cupy",
    "pyclesperanto",
    "pyclesperanto (cuda)",
    "pyclesperanto (metal)",
]

BACKEND_COLORS = {
    "numpy": "#8c8c8c",
    "cupy": "#880086",
    "pyclesperanto": "#e34a4a",
    "pyclesperanto (cuda)": "#2ca02c",
    "pyclesperanto (metal)": "#1f9fdf",
}

def shorten_backend_name(backend: str) -> str:
    """Convert backend name to a short, display-friendly form."""
    return BACKEND_SHORT_NAMES.get(backend, backend)


OPERATION_BASE_NAMES = {
    "elementwise_simple": "Elementwise (simple)",
    "elementwise": "Elementwise",
    "gaussian": "Gaussian blur",
    "convolve": "Convolution",
    "matmul": "Matrix mult.",
    "sum": "Sum reduction",
    "std": "Std. deviation",
    "fft": "FFT",
    "slicing": "Slicing",
}

# How to render the parenthesized parameter for a given base operation,
# e.g. "gaussian(7.0)" -> "sigma=7.0", "convolve(7)" -> "7x7 kernel".
OPERATION_PARAM_FORMATTERS = {
    "gaussian": lambda p: f"\u03c3={p}",
    "convolve": lambda p: f"radius={p}",
    "matmul": lambda p: p.upper(),
    "slicing": lambda p: p,
}


def format_operation_label(operation: str) -> str:
    """Convert a raw operation identifier (e.g. 'gaussian(7.0)',
    'matmul (2d)', 'elementwise_simple') into a readable, two-line
    xticklabel such as 'Gaussian blur\\n(sigma=7.0)'."""
    match = re.match(r"^([a-zA-Z_]+)\s*(?:\((.+)\))?$", operation)
    if not match:
        return operation.replace("_", " ").title()

    base, param = match.group(1), match.group(2)
    label = OPERATION_BASE_NAMES.get(base, base.replace("_", " ").title())

    if param is None:
        return label

    formatter = OPERATION_PARAM_FORMATTERS.get(base, lambda p: p)
    return f"{label}\n({formatter(param)})"


def size_label_from_shape(shape: list | tuple | None, fallback: str) -> str:
    """Normalize inconsistent size labels ('128MB', '128MB', ...) using the
    actual array shape, so the same problem size is grouped together
    regardless of which machine/run produced it."""
    if not shape:
        return fallback
    n_elements = 1
    for dim in shape:
        n_elements *= dim
    n_bytes = n_elements * 4  # float32
    n_mib = n_bytes / (1024**2)
    if n_mib >= 1024:
        size_str = f"{n_mib / 1024:.1f} GiB"
    else:
        size_str = f"{n_mib:.0f} MiB"
    shape_str = "x".join(str(d) for d in shape)
    return f"{size_str} ({shape_str})"


def load_all_benchmark_results(file_paths: list[str]) -> pd.DataFrame:
    """
    Load benchmark results from multiple JSON files into a single tidy
    DataFrame, one row per individual measurement.

    Measurements from all files are pooled together; no per-system
    distinction is kept in the resulting data.

    Args:
        file_paths: List of paths to pytest-benchmark JSON output files,
            one file per benchmarked system.

    Returns:
        DataFrame with columns: backend, backend_short, operation, size,
        time_ms.
    """
    all_results = []

    for filepath in file_paths:
        with open(filepath) as f:
            data = json.load(f)

        for benchmark in data.get("benchmarks", []):
            extra_info = benchmark.get("extra_info", {})
            stats = benchmark.get("stats", {})

            backend = extra_info.get("backend", "unknown")
            operation = extra_info.get("operation", "unknown")
            size = size_label_from_shape(
                extra_info.get("size_shape"), extra_info.get("size", "unknown")
            )

            measurements = stats.get("data", [stats.get("mean", 0)])

            for measurement in measurements:
                all_results.append(
                    {
                        "backend": backend,
                        "backend_short": shorten_backend_name(backend),
                        "operation": operation,
                        "size": size,
                        "time_ms": measurement * 1000,
                    }
                )

    return pd.DataFrame(all_results)


def create_boxplot(
    df: pd.DataFrame,
    size: str,
    output_path: str,
) -> None:
    """
    Create a single boxplot for one problem size: operations on the x-axis,
    backends shown as colored, dodged box groups. Results from all systems
    are pooled together per backend/operation.
    """
    subset = df[df["size"] == size].copy()
    if subset.empty:
        print(f"No data found for size: {size}")
        return

    operations = sorted(subset["operation"].unique())
    backends_present = [b for b in BACKEND_ORDER if b in subset["backend"].unique()]

    subset["operation"] = pd.Categorical(
        subset["operation"], categories=operations, ordered=True
    )
    subset["backend"] = pd.Categorical(
        subset["backend"], categories=backends_present, ordered=True
    )

    palette = {b: BACKEND_COLORS.get(b, "#cccccc") for b in backends_present}

    fig, ax = plt.subplots(figsize=(max(10, len(operations) * 1.8), 6.5))

    sns.boxplot(
        data=subset,
        x="operation",
        y="time_ms",
        hue="backend",
        order=operations,
        hue_order=backends_present,
        palette=palette,
        showfliers=False,
        width=0.7,
        linewidth=1.1,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_xticks(range(len(operations)))
    ax.set_xticklabels(
        [format_operation_label(op) for op in operations],
        rotation=20,
        ha="center",
        fontsize=9,
    )
    ax.set_title(f"Benchmark Results — {size}", fontsize=15, fontweight="bold", pad=48)
    ax.grid(True, which="major", axis="y", alpha=0.35)
    ax.grid(True, which="minor", axis="y", alpha=0.15)
    ax.get_legend().remove()
    handles = [
        Patch(facecolor=palette[b], edgecolor="black", linewidth=0.8, label=shorten_backend_name(b))
        for b in backends_present
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(backends_present),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        borderpad=0.7,
        handlelength=1.4,
        handleheight=1.4,
        columnspacing=1.6,
        fontsize=11,
    )
    legend.get_frame().set_linewidth(0.8)

    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Boxplot saved to: {output_path}")


def main():
    """Main entry point for the scientific boxplot generation CLI."""
    input_path = Path(".")
    json_files = list(input_path.rglob("benchmark_*.json"))
    if not json_files:
        print(f"No benchmark JSON files found in {input_path}")
        return

    print(f"Found {len(json_files)} benchmark files")

    df = load_all_benchmark_results([str(f) for f in json_files])
    if df.empty:
        print("No benchmark data found")
        return

    print(f"Loaded {len(df)} measurements across {df['operation'].nunique()} operations")
    print(f"Backends: {', '.join(sorted(df['backend'].unique()))}")
    print(f"Sizes: {', '.join(sorted(df['size'].unique()))}")

    output_dir = Path("scientific_boxplots")
    output_dir.mkdir(exist_ok=True)

    for size in sorted(df["size"].unique()):
        safe_name = re.sub(r"[^\w.-]+", "_", size).strip("_")
        create_boxplot(
            df,
            size=size,
            output_path=str(output_dir / f"boxplot_{safe_name}.png"),
        )

    print(f"\nBoxplots saved to: {output_dir}")


if __name__ == "__main__":
    main()
