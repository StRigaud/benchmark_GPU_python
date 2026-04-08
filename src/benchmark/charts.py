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
    colors = {"numpy": "#bababa", "cupy": "#880086", "pyclesperanto": "#ff0000ff", "pyclesperanto (cuda)": "#00b700", "pyclesperanto (metal)": "#00bfffff"}
    
    # Create one chart per size
    for size in sizes:
        size_data = df[df["size"] == size]
        
        if size_data.empty:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        group_spacing = max(len(backends) * 0.25 + 0.4, 1.0)
        x_positions = [i * group_spacing for i in range(len(operations))]
        width = 0.25
        
        for i, backend in enumerate(backends):
            backend_data = size_data[size_data["backend"] == backend]
            
            if backend_data.empty:
                continue
            
            means = []
            for operation in operations:
                op_data = backend_data[backend_data["operation"] == operation]
                if not op_data.empty:
                    means.append(op_data["mean"].values[0] * 1000)  # Convert to ms
                else:
                    # Use NaN for missing values to avoid issues with log scale
                    means.append(float('nan'))
            
            offset = (i - len(backends) / 2 + 0.5) * width
            color = colors.get(backend, f"C{i}")
            bars = ax.bar(
                [x + offset for x in x_positions],
                means,
                width,
                label=backend,
                color=color,
            )
            # Annotate values on top of bars in matching color
            for bar in bars:
                height = bar.get_height()
                if height and not pd.isna(height) and height > 0:
                    # Small offset (5%) to sit above the bar; works with log scale
                    offset_y = height * 0.05
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + offset_y,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        color=bar.get_facecolor(),
                        fontsize=7
                    )
        
        ax.set_xlabel("Operation")
        ax.set_ylabel("Time (ms)")
        # Use log scale for readability across wide performance ranges
        ax.set_yscale("log")
        ax.set_title(f"{title} - Size: {size} - Time in ms (log scale)")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([op.replace('_', ' ').title() for op in operations],
                           rotation=45, ha="right")
        ax.legend(frameon=False)
        ax.yaxis.set_visible(False)
        ax.grid(False)
        # Make plots spineless for a cleaner visual
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
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
    # Plot only non-baseline backends; keep baseline as reference line
    backends = [b for b in sorted(df["backend"].unique()) if b != baseline]
    
    # Color mapping for backends
    colors = {"cupy": "#880086", "pyclesperanto": "#ff0000ff", "pyclesperanto (cuda)": "#00b700", "pyclesperanto (metal)": "#00bfffff"}
    
    # Create one chart per size
    for size in sizes:
        size_data = df[df["size"] == size]
        
        if size_data.empty:
            continue
        
        # Calculate speedups relative to baseline (baseline = 0)
        speedup_data = []
        for operation in operations:
            baseline_data = size_data[(size_data["operation"] == operation) & 
                                      (size_data["backend"] == baseline)]
            if baseline_data.empty:
                continue
            baseline_time = baseline_data["mean"].values[0]
            # Other backends: speedup minus 1 (relative to baseline)
            for backend in backends:
                backend_data = size_data[(size_data["operation"] == operation) & 
                                         (size_data["backend"] == backend)]
                if not backend_data.empty:
                    raw_speedup = baseline_time / backend_data["mean"].values[0]
                    rel_speedup = raw_speedup - 1.0
                    speedup_data.append({
                        "operation": operation,
                        "backend": backend,
                        "speedup": rel_speedup
                    })
        
        if not speedup_data:
            continue
        
        speedup_df = pd.DataFrame(speedup_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        group_spacing = max(len(backends) * 0.35 + 0.4, 1.0)
        x_positions = [i * group_spacing for i in range(len(operations))]
        width = 0.35
        
        for i, backend in enumerate(backends):
            backend_speedups = []
            for operation in operations:
                data = speedup_df[(speedup_df["operation"] == operation) & 
                                  (speedup_df["backend"] == backend)]
                if not data.empty:
                    backend_speedups.append(data["speedup"].values[0])
                else:
                    backend_speedups.append(float('nan'))
            
            offset = (i - len(backends) / 2 + 0.5) * width
            color = colors.get(backend, f"C{i}")
            bars = ax.bar([x + offset for x in x_positions], backend_speedups, width,
                          label=backend, color=color)
            # Annotate speedup values; place below if negative
            for bar in bars:
                height = bar.get_height()
                if height and not pd.isna(height):
                    offset_y = (abs(height) + 1e-6) * 0.03
                    x = bar.get_x() + bar.get_width() / 2
                    if height >= 0:
                        y = height + offset_y
                        va = "bottom"
                    else:
                        y = height - offset_y
                        va = "top"
                    ax.text(
                        x,
                        y,
                        f"{height:.2f}",
                        ha="center",
                        va=va,
                        color=bar.get_facecolor(),
                        fontsize=7
                    ) 
        
        # Baseline at 0
        ax.set_yscale("symlog", linthresh=1)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7, 
                   label=f"{baseline} baseline")
        ax.set_xlabel("Operation")
        ax.set_ylabel(f"Relative Speedup vs {baseline} (x - 1)")
        ax.set_title(f"GPU Speedup Comparison - Size: {size} - Relative to {baseline} (log scale)")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels([op.replace('_', ' ').title() for op in operations],
                           rotation=45, ha="right")
        ax.legend(frameon=False)
        ax.yaxis.set_visible(False)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        
        base_path = Path(output_path)
        size_output = base_path.parent / f"{base_path.stem}_{size}{base_path.suffix}"
        plt.tight_layout()
        plt.savefig(size_output, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Speedup chart saved to: {size_output}")


def main():
    """Main entry point for chart generation CLI."""

    # Find benchmark results
    input_path = Path(".")
    if input_path.is_dir():

        # find all JSON files that match the pattern "*_benchmark_*.json" 
        json_files = list(input_path.rglob("*_benchmark_*.json"))
        if len(json_files) == 0:
            print(f"No benchmark JSON files found in {input_path}")
            return
        # json_file = max(json_files, key=os.path.getmtime)
    # else:
    #     json_file = input_path
    
    print(f"Found {len(json_files)} benchmark files in {input_path}")
    
    for json_file in json_files:

        print(f"\nProcessing benchmark file: {json_file}")

        # Create output directory from json filename
        output_dir = Path(json_file).parent / f"{Path(json_file).stem}_charts"
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
