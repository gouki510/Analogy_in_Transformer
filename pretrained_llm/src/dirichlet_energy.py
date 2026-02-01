"""
Dirichlet Energy calculation for analogy graphs.

Dirichlet energy measures the smoothness of embeddings over a graph structure.
For analogy tasks, it quantifies how similar the embeddings of analogically
related entities are.

E(G) = sum_{(i,j) in edges} ||x_i - x_j||^2

where x_i is the embedding of node i.

Normalized dirichlet energy: sum_{(i,j) in edges} ||x_i - x_j||^2 / mean_{t in tokens}(||x_t||^2)
    This normalization accounts for different embedding scales across layers.
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class DirichletResult:
    """Result of Dirichlet energy calculation."""
    energy: float  # Sum of squared L2 distances
    layer: int
    method: str  # 'mean' or 'proxy' or 'position'
    edge_energies: dict[str, float]  # Individual edge contributions


def compute_dirichlet_energy(
    node_embeddings: dict[str, torch.Tensor],
    edges: list[tuple[str, str]],
) -> tuple[float, dict[str, float]]:
    """
    Compute Dirichlet energy for a graph.

    Args:
        node_embeddings: Dictionary mapping node names to embedding vectors
        edges: List of (source, target) tuples representing directed edges

    Returns:
        total_energy: Sum of squared distances over all edges
        edge_energies: Dictionary mapping edge string to its energy contribution
    """
    total_energy = 0.0
    edge_energies = {}

    for source, target in edges:
        if source not in node_embeddings:
            raise ValueError(f"Source node '{source}' not found in embeddings")
        if target not in node_embeddings:
            raise ValueError(f"Target node '{target}' not found in embeddings")

        x_source = node_embeddings[source]
        x_target = node_embeddings[target]

        # ||x_source - x_target||^2
        diff = x_source - x_target
        edge_energy = (diff ** 2).sum().item()

        edge_key = f"{source}->{target}"
        edge_energies[edge_key] = edge_energy
        total_energy += edge_energy

    return total_energy, edge_energies


def compute_dirichlet_energy_all_layers(
    extractor,  # EmbeddingExtractor
    embeddings: torch.Tensor,
    token_info: list,
    edges: list[tuple[str, str]],
    method: str = "mean",
    proxy_tokens: dict[str, str] | None = None,
) -> list[DirichletResult]:
    """
    Compute Dirichlet energy for all layers.

    Args:
        extractor: EmbeddingExtractor instance
        embeddings: Full embedding tensor (n_layers + 1, seq_len, d_model)
        token_info: Token information from tokenization
        edges: List of (source, target) node pairs
        method: 'mean' or 'proxy'
        proxy_tokens: If method='proxy', dict mapping node names to proxy tokens

    Returns:
        List of DirichletResult for each layer
    """
    n_layers = embeddings.shape[0]
    results = []

    for layer in range(n_layers):
        # Extract node embeddings for this layer
        node_embeddings = {}

        for source, target in edges:
            for node in [source, target]:
                if node not in node_embeddings:
                    if method == "mean":
                        node_embeddings[node] = extractor.get_node_embedding(
                            embeddings, token_info, node, layer, method="mean"
                        )
                    else:
                        proxy_token = proxy_tokens.get(node) if proxy_tokens else None
                        node_embeddings[node] = extractor.get_node_embedding(
                            embeddings, token_info, node, layer,
                            method="proxy", proxy_token=proxy_token
                        )

        # Compute Dirichlet energy
        energy, edge_energies = compute_dirichlet_energy(node_embeddings, edges)

        results.append(DirichletResult(
            energy=energy,
            layer=layer,
            method=method,
            edge_energies=edge_energies,
        ))

    return results


def normalize_dirichlet_energy(
    results: list[DirichletResult],
    embeddings: torch.Tensor,
) -> list[float]:
    """
    Normalize Dirichlet energy by the average embedding norm.

    This helps compare energies across layers with different embedding scales.

    Args:
        results: List of DirichletResult
        embeddings: Full embedding tensor

    Returns:
        List of normalized energies
    """
    normalized = []

    for result in results:
        layer = result.layer
        # Average squared norm of embeddings at this layer
        layer_embeddings = embeddings[layer]  # (seq_len, d_model)
        avg_norm_sq = (layer_embeddings ** 2).sum(dim=-1).mean().item()

        # Normalize by average norm squared
        if avg_norm_sq > 0:
            normalized.append(result.energy / avg_norm_sq)
        else:
            normalized.append(0.0)

    return normalized


def compute_dirichlet_energy_with_positions(
    extractor,  # EmbeddingExtractor
    embeddings: torch.Tensor,
    token_info: list,
    edges: list[tuple[str, str]],
    node_positions: dict[str, dict],
) -> list[DirichletResult]:
    """
    Compute Dirichlet energy using specified token positions.

    This method allows precise control over which token occurrence to use
    for each node, which is important when tokens appear multiple times.

    Args:
        extractor: EmbeddingExtractor instance
        embeddings: Full embedding tensor (n_layers + 1, seq_len, d_model)
        token_info: Token information from tokenization
        edges: List of (source, target) node pairs
        node_positions: Dict mapping node names to position config:
            {
                "<e1>": {"proxy_token": "1", "occurrence": -1},
                "<e6>": {"proxy_token": "6", "occurrence": -1},
                ...
            }
            occurrence: -1 for last, 0 for first, etc.

    Returns:
        List of DirichletResult for each layer
    """
    n_layers = embeddings.shape[0]
    results = []

    # Pre-compute all token positions
    token_occurrences = {}
    for node, config in node_positions.items():
        proxy_token = config["proxy_token"]
        if proxy_token not in token_occurrences:
            token_occurrences[proxy_token] = extractor.find_all_occurrences(
                token_info, proxy_token
            )

    # Print position information
    print("\nUsing token positions:")
    for node, config in node_positions.items():
        proxy_token = config["proxy_token"]
        occurrence = config["occurrence"]
        positions = token_occurrences[proxy_token]

        # Handle negative indexing
        actual_idx = occurrence if occurrence >= 0 else len(positions) + occurrence
        actual_pos = positions[actual_idx] if 0 <= actual_idx < len(positions) else "ERROR"

        print(f"  {node}: token '{proxy_token}' occurrence {occurrence} -> position {actual_pos}")

    for layer in range(n_layers):
        node_embeddings = {}

        for source, target in edges:
            for node in [source, target]:
                if node not in node_embeddings:
                    config = node_positions[node]
                    node_embeddings[node] = extractor.get_node_embedding_by_occurrence(
                        embeddings,
                        token_info,
                        proxy_token=config["proxy_token"],
                        occurrence=config["occurrence"],
                        layer=layer,
                    )

        energy, edge_energies = compute_dirichlet_energy(node_embeddings, edges)

        results.append(DirichletResult(
            energy=energy,
            layer=layer,
            method="position",
            edge_energies=edge_energies,
        ))

    return results


def compute_dirichlet_energy_with_mean_positions(
    extractor,  # EmbeddingExtractor
    embeddings: torch.Tensor,
    token_info: list,
    edges: list[tuple[str, str]],
    node_positions: dict[str, dict],
) -> list[DirichletResult]:
    """
    Compute Dirichlet energy using mean embedding of all tokens in each entity.

    This method uses the mean embedding of all tokens that make up an entity
    (e.g., '<', 'e', '1', '>') rather than a single proxy token.

    Args:
        extractor: EmbeddingExtractor instance
        embeddings: Full embedding tensor (n_layers + 1, seq_len, d_model)
        token_info: Token information from tokenization
        edges: List of (source, target) node pairs
        node_positions: Dict mapping node names to position config:
            {
                "<e1>": {"occurrence": -1},  # -1 for last occurrence
                "<e6>": {"occurrence": -1},
                ...
            }

    Returns:
        List of DirichletResult for each layer
    """
    n_layers = embeddings.shape[0]
    results = []

    # Print position information
    print("\nUsing mean embedding positions:")
    for node, config in node_positions.items():
        occurrence = config.get("occurrence", -1)
        all_occurrences = extractor.find_all_node_occurrences(token_info, node)
        n_occurrences = len(all_occurrences)

        # Handle negative indexing
        actual_idx = occurrence if occurrence >= 0 else n_occurrences + occurrence
        if 0 <= actual_idx < n_occurrences:
            positions = all_occurrences[actual_idx]
            tokens_str = [token_info[p].token_str for p in positions]
            print(f"  {node}: occurrence {occurrence} -> positions {positions} (tokens: {tokens_str})")
        else:
            print(f"  {node}: occurrence {occurrence} -> ERROR (only {n_occurrences} occurrences)")

    for layer in range(n_layers):
        node_embeddings = {}

        for source, target in edges:
            for node in [source, target]:
                if node not in node_embeddings:
                    config = node_positions[node]
                    occurrence = config.get("occurrence", -1)
                    node_embeddings[node] = extractor.get_node_embedding_by_occurrence_mean(
                        embeddings,
                        token_info,
                        node_pattern=node,
                        occurrence=occurrence,
                        layer=layer,
                    )

        energy, edge_energies = compute_dirichlet_energy(node_embeddings, edges)

        results.append(DirichletResult(
            energy=energy,
            layer=layer,
            method="mean_position",
            edge_energies=edge_energies,
        ))

    return results


def print_dirichlet_results(results: list[DirichletResult], normalized: list[float] | None = None):
    """Pretty print Dirichlet energy results."""
    print("\n" + "=" * 60)
    print("Dirichlet Energy Analysis")
    print("=" * 60)

    for i, result in enumerate(results):
        layer_name = "Embedding" if result.layer == 0 else f"Layer {result.layer - 1}"
        norm_str = f" (normalized: {normalized[i]:.6f})" if normalized else ""

        print(f"\n{layer_name}: E = {result.energy:.6f}{norm_str}")
        for edge, energy in result.edge_energies.items():
            print(f"  {edge}: {energy:.6f}")

    print("\n" + "=" * 60)
