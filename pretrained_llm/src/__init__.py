"""
Source code for pretrained LLM analogy analysis.
"""

from .embeddings import EmbeddingExtractor, load_sample, print_tokenization
from .dirichlet_energy import (
    compute_dirichlet_energy_all_layers,
    compute_dirichlet_energy_with_positions,
    compute_dirichlet_energy_with_mean_positions,
    normalize_dirichlet_energy,
    print_dirichlet_results,
)
from .functor_similarity import (
    compute_functor_similarity_all_layers,
    print_functor_similarity_results,
)
from .logit_lens import (
    compute_logit_lens_all_layers,
    compute_correlation_with_metrics,
    print_logit_lens_results,
)

__all__ = [
    "EmbeddingExtractor",
    "load_sample",
    "print_tokenization",
    "compute_dirichlet_energy_all_layers",
    "compute_dirichlet_energy_with_positions",
    "compute_dirichlet_energy_with_mean_positions",
    "normalize_dirichlet_energy",
    "print_dirichlet_results",
    "compute_functor_similarity_all_layers",
    "print_functor_similarity_results",
    "compute_logit_lens_all_layers",
    "compute_correlation_with_metrics",
    "print_logit_lens_results",
]
