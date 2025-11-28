"""
Chart generation module for benchmark results visualization.

This module generates comparison charts from pytest-benchmark results.
"""

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def load_benchmark_results(filepath: str) -> pd.DataFrame:
    """
    Load benchmark results from a JSON file.
    
    Args:
        filepath: Path to the pytest-benchmark JSON output file.
    
    Returns:
        DataFrame containing benchmark results.
    """
    with open(filepath) as f:
        data = json.load(f)
    
    results = []
    for benchmark in data["benchmarks"]:
        results.append({
            "name": benchmark["name"],
            "mean": benchmark["stats"]["mean"],
            "stddev": benchmark["stats"]["stddev"],
            "min": benchmark["stats"]["min"],
            "max": benchmark["stats"]["max"],
            "rounds": benchmark["stats"]["rounds"],
        })
    
    return pd.DataFrame(results)


def parse_benchmark_name(name: str) -> tuple:
    """
    Parse benchmark name to extract operation and backend.
    
    Args:
        name: Benchmark test name (e.g., "test_add[numpy-small]")
    
    Returns:
        Tuple of (operation, backend, size)
    """
    # Expected format: test_operation[backend-size]
    if "[" in name and "]" in name:
        base = name.split("[")[0].replace("test_", "")
        params = name.split("[")[1].rstrip("]")
        parts = params.split("-")
        if len(parts) >= 2:
            return base, parts[0], parts[1]
    return name, "unknown", "unknown"


def create_comparison_chart(
    df: pd.DataFrame,
    output_path: str = "benchmark_comparison.png",
    title: str = "Benchmark Comparison"
) -> None:
    """
    Create a bar chart comparing benchmark results across backends.
    
    Args:
        df: DataFrame with benchmark results.
        output_path: Path to save the chart.
        title: Chart title.
    """
    # Parse benchmark names
    df["operation"] = df["name"].apply(lambda x: parse_benchmark_name(x)[0])
    df["backend"] = df["name"].apply(lambda x: parse_benchmark_name(x)[1])
    df["size"] = df["name"].apply(lambda x: parse_benchmark_name(x)[2])
    
    # Create pivot table for plotting
    operations = df["operation"].unique()
    backends = df["backend"].unique()
    sizes = df["size"].unique()
    
    # Create figure with subplots for each operation
    n_ops = len(operations)
    fig, axes = plt.subplots(1, max(n_ops, 1), figsize=(6 * max(n_ops, 1), 6))
    
    if n_ops == 1:
        axes = [axes]
    
    colors = {"numpy": "#1f77b4", "cupy": "#ff7f0e", "pyclesperanto": "#2ca02c"}
    
    for idx, op in enumerate(operations):
        ax = axes[idx]
        op_data = df[df["operation"] == op]
        
        x_positions = range(len(sizes))
        width = 0.25
        
        for i, backend in enumerate(backends):
            backend_data = op_data[op_data["backend"] == backend]
            if not backend_data.empty:
                means = []
                stds = []
                for size in sizes:
                    size_data = backend_data[backend_data["size"] == size]
                    if not size_data.empty:
                        means.append(size_data["mean"].values[0] * 1000)  # Convert to ms
                        stds.append(size_data["stddev"].values[0] * 1000)
                    else:
                        means.append(0)
                        stds.append(0)
                
                offset = (i - len(backends) / 2 + 0.5) * width
                color = colors.get(backend, f"C{i}")
                ax.bar(
                    [x + offset for x in x_positions],
                    means,
                    width,
                    yerr=stds,
                    label=backend,
                    color=color,
                    capsize=3
                )
        
        ax.set_xlabel("Array Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{op.replace('_', ' ').title()}")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(sizes)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Chart saved to: {output_path}")


def create_speedup_chart(
    df: pd.DataFrame,
    baseline: str = "numpy",
    output_path: str = "speedup_comparison.png"
) -> None:
    """
    Create a speedup chart showing performance relative to baseline.
    
    Args:
        df: DataFrame with benchmark results.
        baseline: Baseline backend for comparison.
        output_path: Path to save the chart.
    """
    # Parse benchmark names
    df["operation"] = df["name"].apply(lambda x: parse_benchmark_name(x)[0])
    df["backend"] = df["name"].apply(lambda x: parse_benchmark_name(x)[1])
    df["size"] = df["name"].apply(lambda x: parse_benchmark_name(x)[2])
    
    operations = df["operation"].unique()
    backends = [b for b in df["backend"].unique() if b != baseline]
    sizes = df["size"].unique()
    
    # Calculate speedups
    speedup_data = []
    for op in operations:
        for size in sizes:
            baseline_data = df[(df["operation"] == op) & 
                               (df["backend"] == baseline) & 
                               (df["size"] == size)]
            if baseline_data.empty:
                continue
            baseline_time = baseline_data["mean"].values[0]
            
            for backend in backends:
                backend_data = df[(df["operation"] == op) & 
                                   (df["backend"] == backend) & 
                                   (df["size"] == size)]
                if not backend_data.empty:
                    speedup = baseline_time / backend_data["mean"].values[0]
                    speedup_data.append({
                        "operation": op,
                        "backend": backend,
                        "size": size,
                        "speedup": speedup
                    })
    
    if not speedup_data:
        print("No speedup data available")
        return
    
    speedup_df = pd.DataFrame(speedup_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"cupy": "#ff7f0e", "pyclesperanto": "#2ca02c"}
    x_labels = []
    x = 0
    width = 0.35
    
    for op in operations:
        for size in sizes:
            x_labels.append(f"{op}\n({size})")
            for i, backend in enumerate(backends):
                data = speedup_df[(speedup_df["operation"] == op) & 
                                   (speedup_df["backend"] == backend) & 
                                   (speedup_df["size"] == size)]
                if not data.empty:
                    offset = (i - len(backends) / 2 + 0.5) * width
                    color = colors.get(backend, f"C{i}")
                    ax.bar(x + offset, data["speedup"].values[0], width, 
                          label=backend if x == 0 else "", color=color)
            x += 1
    
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, label=f"{baseline} baseline")
    ax.set_xlabel("Operation (Size)")
    ax.set_ylabel(f"Speedup vs {baseline}")
    ax.set_title(f"GPU Speedup Comparison (baseline: {baseline})")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Speedup chart saved to: {output_path}")


def main():
    """Main entry point for chart generation CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison charts"
    )
    parser.add_argument(
        "--input", "-i",
        default=".benchmarks",
        help="Path to benchmark results directory or JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_results",
        help="Output directory for charts"
    )
    
    args = parser.parse_args()
    
    # Find benchmark results
    input_path = Path(args.input)
    if input_path.is_dir():
        # Find most recent benchmark JSON
        json_files = list(input_path.rglob("*.json"))
        if not json_files:
            print(f"No benchmark JSON files found in {input_path}")
            return
        json_file = max(json_files, key=os.path.getmtime)
    else:
        json_file = input_path
    
    print(f"Loading benchmark results from: {json_file}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load and process results
    df = load_benchmark_results(str(json_file))
    
    # Generate charts
    create_comparison_chart(
        df,
        output_path=str(output_dir / "benchmark_comparison.png")
    )
    
    create_speedup_chart(
        df,
        output_path=str(output_dir / "speedup_comparison.png")
    )
    
    print(f"\nCharts saved to: {output_dir}")


if __name__ == "__main__":
    main()
