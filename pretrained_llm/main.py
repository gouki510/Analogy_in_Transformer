"""
Main analysis script for analogy task metrics.

This script computes:
1. Dirichlet energy across layers (with mean and proxy methods)
2. Functor similarity across layers (cosine and spectrum)
3. Logit lens predictions
4. Correlations between metrics and prediction probability

Usage:
    python main.py --sample samples/sample_1.json --device cpu
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.embeddings import EmbeddingExtractor, load_sample, print_tokenization
from src.dirichlet_energy import (
    compute_dirichlet_energy_all_layers,
    compute_dirichlet_energy_with_positions,
    compute_dirichlet_energy_with_mean_positions,
    normalize_dirichlet_energy,
    print_dirichlet_results,
)
from src.functor_similarity import (
    compute_functor_similarity_all_layers,
    print_functor_similarity_results,
)
from src.logit_lens import (
    compute_logit_lens_all_layers,
    compute_correlation_with_metrics,
    print_logit_lens_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze analogy task metrics")
    parser.add_argument(
        "--sample",
        type=str,
        default="samples/sample_1.json",
        help="Path to sample JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="Model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots",
    )
    parser.add_argument(
        "--proxy_method",
        type=str,
        default="position",
        choices=["position", "mean"],
        help="Proxy method: 'position' uses single token, 'mean' uses mean of all tokens in entity",
    )
    parser.add_argument(
        "--no_rmsnorm",
        action="store_true",
        help="Disable RMSNorm application to embeddings (use raw residual stream)",
    )
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def run_analysis(
    sample_path: str,
    model_name: str,
    device: str,
    dtype: torch.dtype,
    output_dir: str,
    generate_plots: bool,
    proxy_method: str = "position",
    apply_rmsnorm: bool = True,
):
    """Run the full analysis pipeline."""

    # Load sample
    print(f"\nLoading sample from {sample_path}...")
    sample = load_sample(sample_path)
    prompt = sample["prompt"]
    target_token = sample["target_token"]
    sample_name = sample.get("name", Path(sample_path).stem)

    # Create sample-specific output directory
    # Structure: results/<sample_name>/<rmsnorm|no_rmsnorm>/<proxy_method>
    norm_dir = "rmsnorm" if apply_rmsnorm else "no_rmsnorm"
    proxy_dir = "mean_proxy" if proxy_method == "mean" else "num_proxy"
    output_path = Path(output_dir) / sample_name / norm_dir / proxy_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Sample: {sample_name}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Target token: '{target_token}'")
    print(f"Proxy method: {proxy_method}")
    print(f"Apply RMSNorm: {apply_rmsnorm}")
    print(f"Output directory: {output_path}")

    # Initialize extractor and load model
    extractor = EmbeddingExtractor(
        model_name=model_name,
        device=device,
        dtype=dtype,
    )
    extractor.load_model()

    # Extract embeddings
    print("\nExtracting embeddings from all layers...")
    embeddings, token_info = extractor.extract_all_layer_embeddings(prompt, apply_ln=apply_rmsnorm)
    print(f"Embeddings shape: {embeddings.shape}")

    # Print tokenization
    print_tokenization(token_info)

    # Parse graph edges from sample
    edges = [(e["source"], e["target"]) for e in sample["graph"]["edges"]]
    print(f"\nGraph edges (functor relations): {edges}")

    # ==========================================================================
    # 1. Dirichlet Energy Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. DIRICHLET ENERGY ANALYSIS")
    print("=" * 70)

    # Check if sample has explicit node_positions (for correct position handling)
    if "node_positions" in sample:
        if proxy_method == "mean":
            print("\n--- Method: Mean Embedding (using functor relation context) ---")
            dirichlet_results = compute_dirichlet_energy_with_mean_positions(
                extractor, embeddings, token_info, edges, sample["node_positions"]
            )
        else:
            print("\n--- Method: Position-based (using functor relation context) ---")
            dirichlet_results = compute_dirichlet_energy_with_positions(
                extractor, embeddings, token_info, edges, sample["node_positions"]
            )
        normalized_results = normalize_dirichlet_energy(dirichlet_results, embeddings)
        print_dirichlet_results(dirichlet_results, normalized_results)

        dirichlet_primary = dirichlet_results
        normalized_primary = normalized_results
    else:
        # Fallback to mean method if no positions specified
        print("\n--- Method: Mean Embedding ---")
        dirichlet_mean = compute_dirichlet_energy_all_layers(
            extractor, embeddings, token_info, edges, method="mean"
        )
        normalized_mean = normalize_dirichlet_energy(dirichlet_mean, embeddings)
        print_dirichlet_results(dirichlet_mean, normalized_mean)

        dirichlet_primary = dirichlet_mean
        normalized_primary = normalized_mean

    # ==========================================================================
    # 2. Functor Similarity Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. FUNCTOR SIMILARITY ANALYSIS")
    print("=" * 70)

    # Count functor occurrences
    functor_token = "~"
    n_functors = sum(1 for info in token_info if functor_token in info.token_str)
    print(f"\nFound {n_functors} occurrences of functor token '{functor_token}'")

    if n_functors >= 2:
        functor_results = compute_functor_similarity_all_layers(
            extractor, embeddings, token_info, functor_token, n_functors
        )
        print_functor_similarity_results(functor_results)
    else:
        print("Warning: Need at least 2 functors for similarity analysis")
        functor_results = None

    # ==========================================================================
    # 3. Logit Lens Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. LOGIT LENS ANALYSIS")
    print("=" * 70)

    logit_lens_results = compute_logit_lens_all_layers(
        extractor.model, embeddings, target_token, position=-1, top_k=10
    )
    print_logit_lens_results(logit_lens_results, target_token)

    # ==========================================================================
    # 4. Correlation Analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. CORRELATION ANALYSIS")
    print("=" * 70)

    if functor_results:
        correlations = compute_correlation_with_metrics(
            logit_lens_results, dirichlet_primary, functor_results
        )

        print("\nCorrelation between target token probability and:")
        print(f"  Dirichlet Energy: {correlations['dirichlet_energy']:.4f}")
        print(f"  Cosine Similarity: {correlations['cosine_similarity']:.4f}")
        print(f"  Spectrum Ratio: {correlations['spectrum_ratio']:.4f}")

    # ==========================================================================
    # 5. Save Results
    # ==========================================================================
    results = {
        "sample": sample_path,
        "model": model_name,
        "prompt": prompt,
        "target_token": target_token,
        "n_layers": embeddings.shape[0] - 1,
        "dirichlet_energy": [
            {"layer": r.layer, "energy": r.energy, "normalized": n, "edge_energies": r.edge_energies}
            for r, n in zip(dirichlet_primary, normalized_primary)
        ],
        "logit_lens": [
            {
                "layer": r.layer,
                "target_prob": r.target_token_prob,
                "target_logit": r.target_token_logit,
                "target_logit_relative": r.target_token_logit_relative,
                "log_prob": r.log_prob,
                "target_rank": r.target_token_rank,
            }
            for r in logit_lens_results
        ],
    }

    if functor_results:
        results["correlations"] = correlations

    results_file = output_path / "analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # ==========================================================================
    # 6. Generate Plots
    # ==========================================================================
    if generate_plots:
        generate_analysis_plots(
            results, dirichlet_primary, functor_results,
            logit_lens_results, normalized_primary, output_path
        )


def setup_plot_style():
    """Setup matplotlib style for publication-quality plots."""
    plt.close()
    plt.rcParams.update({
        # Font - Large size for readability
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 28,

        # Axes - Large labels
        'axes.labelsize': 32,
        'axes.titlesize': 28,
        'axes.linewidth': 2.0,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks - Large size
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 8,
        'ytick.major.size': 8,

        # Legend - Larger size
        'legend.fontsize': 24,
        'legend.frameon': False,

        # Grid
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': ':',

        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        
        # Lines - Thicker
        'lines.linewidth': 3,
        'lines.markersize': 10,
    })
    plt.close()
    plt.style.use("ggplot")
    plt.rcParams.update({'axes.grid': True})


def generate_analysis_plots(
    results, dirichlet_results, functor_results,
    logit_lens_results, normalized_dirichlet, output_path
):
    """Generate visualization plots."""
    
    # Setup plot style
    setup_plot_style()

    layers = list(range(len(dirichlet_results)))
    target_probs = [r.target_token_prob for r in logit_lens_results]

    # Extract sample info for titles
    prompt = results.get("prompt", "")
    target_token = results.get("target_token", "")
    sequence_str = f"Sequence: {prompt}[{target_token}]"

    # Plot 1: Combined plot - Normalized Dirichlet Energy vs P(target) with dual y-axis
    fig, ax1 = plt.subplots(figsize=(16, 8))

    color1 = '#4C72B0'  # ggplot-compatible blue
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Normalized Dirichlet Energy', color=color1)
    line1 = ax1.plot(layers, normalized_dirichlet, color=color1, marker='o',
                     linewidth=4, markersize=12, label='Normalized Dirichlet Energy')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#C44E52'  # ggplot-compatible red
    ax2.set_ylabel('Target Token Probability', color=color2)
    line2 = ax2.plot(layers, target_probs,
                     color=color2, marker='s', linewidth=4, markersize=12, label='Prob')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.spines['right'].set_visible(True)

    # Combined legend - position at upper left to avoid overlap
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Title with sequence info
    fig.suptitle('Normalized Dirichlet Energy vs Target Token Probability', fontsize=36, y=0.98)
    ax1.set_title(sequence_str, fontsize=20, color='gray', pad=15)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    combined_file = output_path / "dirichlet_vs_probability.pdf"
    plt.savefig(combined_file, bbox_inches='tight')
    print(f"Combined plot saved to {combined_file}")
    plt.close()

    # Plot 2: Dirichlet Energy vs P(target) with dual y-axis (showing correlation)
    fig, ax1 = plt.subplots(figsize=(16, 8))

    dirichlet_energies = [r.energy for r in dirichlet_results]

    # Compute correlation
    corr = np.corrcoef(dirichlet_energies, target_probs)[0, 1]

    color1 = '#4C72B0'
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Dirichlet Energy', color=color1)
    line1 = ax1.plot(layers, dirichlet_energies, color=color1, marker='o',
                     linewidth=4, markersize=12, label='Dirichlet Energy')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#C44E52'
    ax2.set_ylabel('Prob', color=color2)
    line2 = ax2.plot(layers, target_probs,
                     color=color2, marker='s', linewidth=4, markersize=12, label='P')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.spines['right'].set_visible(True)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    fig.suptitle(f'Dirichlet Energy and Prob', fontsize=36, y=0.98)
    ax1.set_title(sequence_str, fontsize=20, color='gray', pad=15)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    corr_file = output_path / "correlation_plot.pdf"
    plt.savefig(corr_file, bbox_inches='tight')
    print(f"Correlation plot saved to {corr_file}")
    plt.close()


def main():
    args = parse_args()

    dtype = get_dtype(args.dtype)

    run_analysis(
        sample_path=args.sample,
        model_name=args.model,
        device=args.device,
        dtype=dtype,
        output_dir=args.output,
        generate_plots=args.plot,
        proxy_method=args.proxy_method,
        apply_rmsnorm=not args.no_rmsnorm,
    )


if __name__ == "__main__":
    main()
