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
        # Extract extra_info if available
        extra_info = benchmark.get("extra_info", {})
        
        results.append({
            "name": benchmark["name"],
            "mean": benchmark["stats"]["mean"],
            "stddev": benchmark["stats"]["stddev"],
            "min": benchmark["stats"]["min"],
            "max": benchmark["stats"]["max"],
            "rounds": benchmark["stats"]["rounds"],
            "size": extra_info.get("size", "unknown"),
            "backend": extra_info.get("backend", "unknown"),
            "operation": extra_info.get("operation", "unknown"),
        })
    
    return pd.DataFrame(results)


def create_comparison_chart(
    df: pd.DataFrame,
    output_path: str = "benchmark_comparison.png",
    title: str = "Benchmark Comparison"
) -> None:
    """
    Create bar charts comparing benchmark results across backends.
    Creates one chart per size, showing all operations with different backends.
    
    Args:
        df: DataFrame with benchmark results.
        output_path: Path to save the chart.
        title: Chart title.
    """
    # Get unique sizes, operations, and backends
    sizes = sorted(df["size"].unique())
    operations = sorted(df["operation"].unique())
    backends = sorted(df["backend"].unique())
    
    # Color mapping for backends
    colors = {"numpy": "#1f77b4", "cupy": "#ff7f0e", "pyclesperanto": "#2ca02c"}
    
    # Create one chart per size
    for size in sizes:
        size_data = df[df["size"] == size]
        
        if size_data.empty:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_positions = range(len(operations))
        width = 0.25
        
        # Plot bars for each backend
        for i, backend in enumerate(backends):
            backend_data = size_data[size_data["backend"] == backend]
            
            if backend_data.empty:
                continue
            
            means = []
            stds = []
            for operation in operations:
                op_data = backend_data[backend_data["operation"] == operation]
                if not op_data.empty:
                    means.append(op_data["mean"].values[0] * 1000)  # Convert to ms
                    stds.append(op_data["stddev"].values[0] * 1000)
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
        
        ax.set_xlabel("Operation")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{title} - Size: {size}")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([op.replace('_', ' ').title() for op in operations])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        # Save chart with size in filename
        base_path = Path(output_path)
        size_output = base_path.parent / f"{base_path.stem}_{size}{base_path.suffix}"
        plt.tight_layout()
        plt.savefig(size_output, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Chart saved to: {size_output}")


def create_speedup_chart(
    df: pd.DataFrame,
    baseline: str = "numpy",
    output_path: str = "speedup_comparison.png"
) -> None:
    """
    Create speedup charts showing performance relative to baseline.
    Creates one chart per size.
    
    Args:
        df: DataFrame with benchmark results.
        baseline: Baseline backend for comparison.
        output_path: Path to save the chart.
    """
    # Get unique sizes, operations, and backends
    sizes = sorted(df["size"].unique())
    operations = sorted(df["operation"].unique())
    backends = [b for b in sorted(df["backend"].unique()) if b != baseline]
    
    # Color mapping for backends
    colors = {"cupy": "#ff7f0e", "pyclesperanto": "#2ca02c"}
    
    # Create one chart per size
    for size in sizes:
        size_data = df[df["size"] == size]
        
        if size_data.empty:
            continue
        
        # Calculate speedups
        speedup_data = []
        for operation in operations:
            baseline_data = size_data[(size_data["operation"] == operation) & 
                                      (size_data["backend"] == baseline)]
            if baseline_data.empty:
                continue
            baseline_time = baseline_data["mean"].values[0]
            
            for backend in backends:
                backend_data = size_data[(size_data["operation"] == operation) & 
                                         (size_data["backend"] == backend)]
                if not backend_data.empty:
                    speedup = baseline_time / backend_data["mean"].values[0]
                    speedup_data.append({
                        "operation": operation,
                        "backend": backend,
                        "speedup": speedup
                    })
        
        if not speedup_data:
            continue
        
        speedup_df = pd.DataFrame(speedup_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_positions = range(len(operations))
        width = 0.35
        
        for i, backend in enumerate(backends):
            backend_speedups = []
            for operation in operations:
                data = speedup_df[(speedup_df["operation"] == operation) & 
                                  (speedup_df["backend"] == backend)]
                if not data.empty:
                    backend_speedups.append(data["speedup"].values[0])
                else:
                    backend_speedups.append(0)
            
            offset = (i - len(backends) / 2 + 0.5) * width
            color = colors.get(backend, f"C{i}")
            ax.bar([x + offset for x in x_positions], backend_speedups, width,
                   label=backend, color=color)
        
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, 
                   label=f"{baseline} baseline")
        ax.set_xlabel("Operation")
        ax.set_ylabel(f"Speedup vs {baseline}")
        ax.set_title(f"GPU Speedup Comparison - Size: {size}")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([op.replace('_', ' ').title() for op in operations])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        # Save chart with size in filename
        base_path = Path(output_path)
        size_output = base_path.parent / f"{base_path.stem}_{size}{base_path.suffix}"
        plt.tight_layout()
        plt.savefig(size_output, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Speedup chart saved to: {size_output}")


def main():
    """Main entry point for chart generation CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison charts"
    )
    parser.add_argument(
        "--input", "-i",
        default=".",
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
