"""
Logit Lens analysis for analogy tasks.

Logit lens applies the unembedding matrix to intermediate layer representations
to see what tokens the model would predict at each layer.

This allows us to track when the correct answer emerges during forward pass.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class LogitLensResult:
    """Result of logit lens analysis at one layer."""
    layer: int
    target_token_prob: float
    target_token_rank: int
    top_tokens: list[tuple[str, float]]  # (token_str, probability)
    target_token_logit: float
    target_token_logit_relative: float  # logit - max_logit (always <= 0)
    log_prob: float  # log(prob), more numerically stable for correlation


def apply_logit_lens(
    model,  # HookedTransformer
    residual: torch.Tensor,
    position: int = -1,
) -> torch.Tensor:
    """
    Apply logit lens to a residual stream vector.

    Args:
        model: TransformerLens model
        residual: Residual stream tensor of shape (seq_len, d_model) or (d_model,)
        position: Which position to analyze (-1 for last)

    Returns:
        Logits tensor of shape (vocab_size,)
    """
    if residual.dim() == 2:
        residual = residual[position]  # (d_model,)

    # Move residual to the same device as the model
    device = model.W_U.device
    residual = residual.to(device)

    # Apply layer norm if the model uses it
    if hasattr(model, 'ln_final'):
        residual = model.ln_final(residual)

    # Apply unembedding: (d_model,) @ (d_model, vocab_size) -> (vocab_size,)
    logits = residual @ model.W_U

    # Add bias if exists
    if model.b_U is not None:
        logits = logits + model.b_U

    return logits


def compute_logit_lens_all_layers(
    model,  # HookedTransformer
    embeddings: torch.Tensor,
    target_token: str,
    position: int = -1,
    top_k: int = 10,
) -> list[LogitLensResult]:
    """
    Apply logit lens at all layers.

    Args:
        model: TransformerLens model
        embeddings: Full embedding tensor (n_layers + 1, seq_len, d_model)
        target_token: The expected correct token (e.g., "7")
        position: Which position to analyze (-1 for last)
        top_k: Number of top tokens to return

    Returns:
        List of LogitLensResult for each layer
    """
    n_layers = embeddings.shape[0]
    results = []

    # Get target token ID - handle multi-token targets by using the first token
    # For multi-digit numbers like "10", the model predicts "1" first, then "0"
    try:
        target_token_id = model.to_single_token(target_token)
    except AssertionError:
        # Multi-token target: use the first token
        tokens = model.to_tokens(target_token, prepend_bos=False)[0]
        target_token_id = tokens[0].item()
        print(f"  Note: '{target_token}' is multi-token, using first token ID: {target_token_id}")

    for layer in range(n_layers):
        residual = embeddings[layer]  # (seq_len, d_model)

        # Apply logit lens
        logits = apply_logit_lens(model, residual, position)

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Get target token probability and logit
        target_prob = probs[target_token_id].item()
        target_logit = logits[target_token_id].item()

        # Compute relative logit (logit - max_logit), always <= 0
        max_logit = logits.max().item()
        target_logit_relative = target_logit - max_logit

        # Compute log probability (more stable for correlation analysis)
        # Use log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob = log_probs[target_token_id].item()

        # Get rank of target token
        sorted_indices = torch.argsort(probs, descending=True)
        target_rank = (sorted_indices == target_token_id).nonzero().item() + 1

        # Get top-k tokens
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = []
        for prob, idx in zip(top_probs, top_indices):
            token_str = model.to_single_str_token(idx.item())
            top_tokens.append((token_str, prob.item()))

        results.append(LogitLensResult(
            layer=layer,
            target_token_prob=target_prob,
            target_token_rank=target_rank,
            top_tokens=top_tokens,
            target_token_logit=target_logit,
            target_token_logit_relative=target_logit_relative,
            log_prob=log_prob,
        ))

    return results


def compute_correlation_with_metrics(
    logit_lens_results: list[LogitLensResult],
    dirichlet_results: list,  # DirichletResult
    functor_results: list,  # FunctorSimilarityResult
) -> dict[str, float]:
    """
    Compute correlation between target token probability and other metrics.

    Args:
        logit_lens_results: Logit lens results
        dirichlet_results: Dirichlet energy results
        functor_results: Functor similarity results

    Returns:
        Dictionary of correlation coefficients
    """
    n = len(logit_lens_results)

    target_probs = np.array([r.target_token_prob for r in logit_lens_results])
    dirichlet_energies = np.array([r.energy for r in dirichlet_results])
    mean_cosines = np.array([r.mean_cosine_similarity for r in functor_results])
    spectrum_ratios = np.array([r.spectrum_ratio for r in functor_results])

    correlations = {}

    # Correlation with Dirichlet energy
    if np.std(dirichlet_energies) > 0 and np.std(target_probs) > 0:
        correlations["dirichlet_energy"] = np.corrcoef(target_probs, dirichlet_energies)[0, 1]
    else:
        correlations["dirichlet_energy"] = 0.0

    # Correlation with cosine similarity
    if np.std(mean_cosines) > 0 and np.std(target_probs) > 0:
        correlations["cosine_similarity"] = np.corrcoef(target_probs, mean_cosines)[0, 1]
    else:
        correlations["cosine_similarity"] = 0.0

    # Correlation with spectrum ratio
    if np.std(spectrum_ratios) > 0 and np.std(target_probs) > 0:
        correlations["spectrum_ratio"] = np.corrcoef(target_probs, spectrum_ratios)[0, 1]
    else:
        correlations["spectrum_ratio"] = 0.0

    return correlations


def print_logit_lens_results(results: list[LogitLensResult], target_token: str):
    """Pretty print logit lens results."""
    print("\n" + "=" * 60)
    print(f"Logit Lens Analysis (target: '{target_token}')")
    print("=" * 60)

    for result in results:
        layer_name = "Embedding" if result.layer == 0 else f"Layer {result.layer - 1}"

        print(f"\n{layer_name}:")
        print(f"  Target '{target_token}' prob: {result.target_token_prob:.6f} (rank {result.target_token_rank})")
        print(f"  Top 5 predictions:")
        for token, prob in result.top_tokens[:5]:
            marker = " <--" if token == target_token else ""
            print(f"    '{token}': {prob:.6f}{marker}")

    print("\n" + "=" * 60)
