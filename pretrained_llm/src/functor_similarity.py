"""
Functor Similarity calculation for analogy tasks.

Functors are the "~" operators representing analogy relations.
This module computes similarity metrics between functor embeddings.

Two metrics:
1. Cosine similarity: pairwise cosine similarity between functor embeddings
2. Spectrum similarity: λ_max(C) / tr(C) where C is the covariance matrix
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class FunctorSimilarityResult:
    """Result of functor similarity calculation."""
    layer: int
    cosine_similarities: dict[str, float]  # Pairwise cosine similarities
    mean_cosine_similarity: float
    spectrum_ratio: float  # λ_max / tr(C)
    eigenvalues: list[float]  # All eigenvalues of covariance matrix


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        x: First vector
        y: Second vector

    Returns:
        Cosine similarity value
    """
    x_norm = torch.norm(x)
    y_norm = torch.norm(y)

    if x_norm == 0 or y_norm == 0:
        return 0.0

    return (torch.dot(x, y) / (x_norm * y_norm)).item()


def compute_spectrum_similarity(functor_embeddings: list[torch.Tensor]) -> tuple[float, list[float]]:
    """
    Compute spectrum similarity for functor embeddings.

    Stack embeddings into matrix X = [f1^T, f2^T, ...]
    Compute covariance C = (1/n) X^T X
    Return λ_max(C) / tr(C)

    Args:
        functor_embeddings: List of functor embedding vectors

    Returns:
        ratio: λ_max / tr(C)
        eigenvalues: List of all eigenvalues
    """
    # Stack into matrix X: (n_functors, d_model)
    X = torch.stack(functor_embeddings, dim=0)
    n_functors = X.shape[0]

    # Covariance matrix C = (1/n) X^T X
    # Shape: (d_model, d_model)
    # Convert to float32 for numerical operations (eigvalsh/trace not implemented for float16/bfloat16)
    X_f32 = X.float()
    C = (1.0 / n_functors) * torch.mm(X_f32.T, X_f32)

    # Compute eigenvalues
    # For large matrices, we only need the largest eigenvalues
    # Use torch.linalg.eigvalsh for symmetric matrices (faster)
    eigenvalues = torch.linalg.eigvalsh(C)

    # Sort in descending order
    eigenvalues = eigenvalues.flip(0)

    # λ_max / tr(C)
    lambda_max = eigenvalues[0].item()
    trace_C = torch.trace(C).item()

    if trace_C == 0:
        ratio = 0.0
    else:
        ratio = lambda_max / trace_C

    return ratio, eigenvalues.tolist()


def compute_functor_similarity(
    functor_embeddings: dict[str, torch.Tensor],
) -> tuple[dict[str, float], float, float, list[float]]:
    """
    Compute all functor similarity metrics.

    Args:
        functor_embeddings: Dictionary mapping functor names to embeddings

    Returns:
        cosine_sims: Pairwise cosine similarities
        mean_cosine: Mean of all pairwise cosine similarities
        spectrum_ratio: λ_max / tr(C)
        eigenvalues: All eigenvalues
    """
    names = list(functor_embeddings.keys())
    embeddings = [functor_embeddings[name] for name in names]

    # Pairwise cosine similarities
    cosine_sims = {}
    all_cosines = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}<->{names[j]}"
            sim = cosine_similarity(embeddings[i], embeddings[j])
            cosine_sims[key] = sim
            all_cosines.append(sim)

    mean_cosine = np.mean(all_cosines) if all_cosines else 0.0

    # Spectrum similarity
    spectrum_ratio, eigenvalues = compute_spectrum_similarity(embeddings)

    return cosine_sims, mean_cosine, spectrum_ratio, eigenvalues


def compute_functor_similarity_all_layers(
    extractor,  # EmbeddingExtractor
    embeddings: torch.Tensor,
    token_info: list,
    functor_token: str,
    n_functors: int,
) -> list[FunctorSimilarityResult]:
    """
    Compute functor similarity for all layers.

    Args:
        extractor: EmbeddingExtractor instance
        embeddings: Full embedding tensor (n_layers + 1, seq_len, d_model)
        token_info: Token information from tokenization
        functor_token: The functor token (e.g., '~')
        n_functors: Number of functor occurrences to analyze

    Returns:
        List of FunctorSimilarityResult for each layer
    """
    n_layers = embeddings.shape[0]
    results = []

    for layer in range(n_layers):
        # Extract functor embeddings for this layer
        functor_embeddings = {}

        for i in range(n_functors):
            name = f"functor_{i + 1}"
            functor_embeddings[name] = extractor.get_functor_embedding(
                embeddings, token_info, functor_token, occurrence=i, layer=layer
            )

        # Compute similarities
        cosine_sims, mean_cosine, spectrum_ratio, eigenvalues = compute_functor_similarity(
            functor_embeddings
        )

        results.append(FunctorSimilarityResult(
            layer=layer,
            cosine_similarities=cosine_sims,
            mean_cosine_similarity=mean_cosine,
            spectrum_ratio=spectrum_ratio,
            eigenvalues=eigenvalues[:10],  # Keep only top 10 eigenvalues
        ))

    return results


def print_functor_similarity_results(results: list[FunctorSimilarityResult]):
    """Pretty print functor similarity results."""
    print("\n" + "=" * 60)
    print("Functor Similarity Analysis")
    print("=" * 60)

    for result in results:
        layer_name = "Embedding" if result.layer == 0 else f"Layer {result.layer - 1}"

        print(f"\n{layer_name}:")
        print(f"  Mean Cosine Similarity: {result.mean_cosine_similarity:.6f}")
        print(f"  Spectrum Ratio (λ_max/tr): {result.spectrum_ratio:.6f}")

        print("  Pairwise Cosine Similarities:")
        for pair, sim in result.cosine_similarities.items():
            print(f"    {pair}: {sim:.6f}")

    print("\n" + "=" * 60)
