"""
Embedding extraction utilities for analogy analysis.
Uses TransformerLens to extract residual stream activations from all layers.
"""

import logging
import torch
import json
from pathlib import Path
from transformer_lens import HookedTransformer
from typing import NamedTuple

# Suppress TransformerLens warnings about LayerNorm centering (Gemma-2 uses RMSNorm, which is fine)
logging.getLogger("root").setLevel(logging.ERROR)


class TokenInfo(NamedTuple):
    """Information about a token's position and content."""
    token_id: int
    token_str: str
    position: int


class EmbeddingExtractor:
    """Extract embeddings from all layers of a transformer model."""

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the embedding extractor.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu', 'cuda', 'mps')
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None

    def load_model(self):
        """Load the model using TransformerLens."""
        print(f"Loading model {self.model_name}...")
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
        )
        print(f"Model loaded. n_layers={self.model.cfg.n_layers}, d_model={self.model.cfg.d_model}")
        return self

    def tokenize(self, prompt: str) -> tuple[torch.Tensor, list[TokenInfo]]:
        """
        Tokenize a prompt and return token information.

        Args:
            prompt: Input text

        Returns:
            tokens: Token IDs tensor
            token_info: List of TokenInfo with position and string for each token
        """
        tokens = self.model.to_tokens(prompt, prepend_bos=True)
        token_strs = self.model.to_str_tokens(prompt, prepend_bos=True)

        token_info = [
            TokenInfo(token_id=tokens[0, i].item(), token_str=token_strs[i], position=i)
            for i in range(len(token_strs))
        ]

        return tokens, token_info

    def find_token_positions(
        self,
        token_info: list[TokenInfo],
        target_tokens: list[str],
    ) -> list[int]:
        """
        Find positions of target tokens in the tokenized sequence.

        Args:
            token_info: List of TokenInfo from tokenization
            target_tokens: List of token strings to find

        Returns:
            List of positions where target tokens appear
        """
        positions = []
        for i, info in enumerate(token_info):
            if info.token_str in target_tokens:
                positions.append(i)
        return positions

    def find_node_positions(
        self,
        token_info: list[TokenInfo],
        node_pattern: str,
    ) -> list[int]:
        """
        Find positions of tokens that make up a node (e.g., '<e1>').

        Args:
            token_info: List of TokenInfo from tokenization
            node_pattern: Pattern like '<e1>' to find

        Returns:
            List of positions for all tokens in the node
        """
        # Build the full tokenized string to find substring
        full_str = "".join([info.token_str for info in token_info])

        # Find where this pattern starts
        pattern_start = full_str.find(node_pattern)
        if pattern_start == -1:
            return []

        # Find which token positions correspond to this pattern
        positions = []
        current_pos = 0
        for i, info in enumerate(token_info):
            token_end = current_pos + len(info.token_str)
            pattern_end = pattern_start + len(node_pattern)

            # Check if this token overlaps with the pattern
            if current_pos < pattern_end and token_end > pattern_start:
                positions.append(i)

            current_pos = token_end

            # Stop if we've passed the pattern
            if current_pos >= pattern_end:
                break

        return positions

    def extract_all_layer_embeddings(
        self,
        prompt: str,
        apply_ln: bool = True,
    ) -> tuple[torch.Tensor, list[TokenInfo]]:
        """
        Extract residual stream embeddings from all layers.

        Args:
            prompt: Input text
            apply_ln: Whether to apply layer norm to embeddings (default True).
                     For each layer i, applies layer i+1's ln1 (or ln_final for last layer).
                     This normalizes the embeddings for meaningful comparison.

        Returns:
            embeddings: Tensor of shape (n_layers + 1, seq_len, d_model)
                       Layer 0 is the embedding layer, layers 1-n are after each transformer block
            token_info: List of TokenInfo for the tokenized prompt
        """
        if self.model is None:
            self.load_model()

        tokens, token_info = self.tokenize(prompt)

        # Run model and cache all residual stream activations
        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=lambda name: "resid_post" in name or name == "hook_embed",
        )

        n_layers = self.model.cfg.n_layers
        seq_len = tokens.shape[1]
        d_model = self.model.cfg.d_model

        # Collect embeddings from all layers
        # Layer 0: embedding layer output
        # Layer i (i > 0): residual stream after layer i-1
        embeddings = torch.zeros(n_layers + 1, seq_len, d_model, dtype=self.dtype)

        # Embedding layer (before any transformer blocks)
        embed = cache["hook_embed"][0]
        if apply_ln:
            # Apply first layer's ln1 to embedding
            embed = self.model.blocks[0].ln1(embed)
        embeddings[0] = embed

        # After each transformer block
        for layer in range(n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0]
            if apply_ln:
                if layer < n_layers - 1:
                    # Apply next layer's ln1
                    resid = self.model.blocks[layer + 1].ln1(resid)
                else:
                    # Apply final layer norm for the last layer
                    resid = self.model.ln_final(resid)
            embeddings[layer + 1] = resid

        return embeddings, token_info

    def get_node_embedding(
        self,
        embeddings: torch.Tensor,
        token_info: list[TokenInfo],
        node_pattern: str,
        layer: int,
        method: str = "mean",
        proxy_token: str | None = None,
    ) -> torch.Tensor:
        """
        Get the embedding for a node at a specific layer.

        Args:
            embeddings: Full embedding tensor (n_layers + 1, seq_len, d_model)
            token_info: Token information
            node_pattern: Pattern like '<e1>'
            layer: Which layer to extract from (0 = embedding, 1+ = after transformer blocks)
            method: 'mean' for average of all tokens, 'proxy' for specific token
            proxy_token: If method='proxy', which token to use (e.g., '1' for '<e1>')

        Returns:
            Embedding vector of shape (d_model,)
        """
        positions = self.find_node_positions(token_info, node_pattern)

        if not positions:
            raise ValueError(f"Could not find node pattern '{node_pattern}' in tokens")

        if method == "mean":
            # Average embedding of all tokens in the node
            node_embeddings = embeddings[layer, positions, :]
            return node_embeddings.mean(dim=0)

        elif method == "proxy":
            # Use a specific token as proxy
            if proxy_token is None:
                raise ValueError("proxy_token must be specified when method='proxy'")

            for pos in positions:
                if token_info[pos].token_str == proxy_token:
                    return embeddings[layer, pos, :]

            raise ValueError(f"Proxy token '{proxy_token}' not found in node '{node_pattern}'")

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_functor_embedding(
        self,
        embeddings: torch.Tensor,
        token_info: list[TokenInfo],
        functor_token: str,
        occurrence: int,
        layer: int,
    ) -> torch.Tensor:
        """
        Get the embedding of a functor token (~) at a specific occurrence.

        Args:
            embeddings: Full embedding tensor
            token_info: Token information
            functor_token: The functor token (e.g., '~')
            occurrence: Which occurrence (0-indexed)
            layer: Which layer to extract from

        Returns:
            Embedding vector of shape (d_model,)
        """
        positions = []
        for i, info in enumerate(token_info):
            if functor_token in info.token_str:
                positions.append(i)

        if occurrence >= len(positions):
            raise ValueError(
                f"Occurrence {occurrence} requested but only {len(positions)} "
                f"occurrences of '{functor_token}' found"
            )

        return embeddings[layer, positions[occurrence], :]

    def find_all_occurrences(
        self,
        token_info: list[TokenInfo],
        target_token: str,
    ) -> list[int]:
        """
        Find all positions where a specific token appears.

        Args:
            token_info: Token information
            target_token: Token string to find (exact match)

        Returns:
            List of positions
        """
        return [i for i, info in enumerate(token_info) if info.token_str == target_token]

    def find_all_node_occurrences(
        self,
        token_info: list[TokenInfo],
        node_pattern: str,
    ) -> list[list[int]]:
        """
        Find all occurrences of a node pattern (e.g., '<e1>') and return their token positions.

        Args:
            token_info: List of TokenInfo from tokenization
            node_pattern: Pattern like '<e1>' to find

        Returns:
            List of lists, where each inner list contains the token positions for one occurrence
            e.g., [[2,3,4,5], [15,16,17,18]] for two occurrences of '<e1>'
        """
        # Build the full tokenized string
        full_str = "".join([info.token_str for info in token_info])

        # Find all occurrences of the pattern
        occurrences = []
        start = 0
        while True:
            pattern_start = full_str.find(node_pattern, start)
            if pattern_start == -1:
                break

            # Find which token positions correspond to this pattern occurrence
            positions = []
            current_pos = 0
            pattern_end = pattern_start + len(node_pattern)

            for i, info in enumerate(token_info):
                token_end = current_pos + len(info.token_str)

                # Check if this token overlaps with the pattern
                if current_pos < pattern_end and token_end > pattern_start:
                    positions.append(i)

                current_pos = token_end

                # Stop if we've passed the pattern
                if current_pos >= pattern_end:
                    break

            if positions:
                occurrences.append(positions)

            # Move past this occurrence
            start = pattern_start + 1

        return occurrences

    def get_node_embedding_by_occurrence_mean(
        self,
        embeddings: torch.Tensor,
        token_info: list[TokenInfo],
        node_pattern: str,
        occurrence: int,
        layer: int,
    ) -> torch.Tensor:
        """
        Get the mean embedding of a node at a specific occurrence.

        This uses all tokens that make up the node (e.g., '<', 'e', '1', '>') and
        averages their embeddings.

        Args:
            embeddings: Full embedding tensor (n_layers + 1, seq_len, d_model)
            token_info: Token information
            node_pattern: Pattern like '<e1>'
            occurrence: Which occurrence (-1 for last, 0 for first, etc.)
            layer: Which layer to extract from

        Returns:
            Mean embedding vector of shape (d_model,)
        """
        all_occurrences = self.find_all_node_occurrences(token_info, node_pattern)

        if not all_occurrences:
            raise ValueError(f"Node pattern '{node_pattern}' not found in sequence")

        # Handle negative indexing (e.g., -1 for last)
        if occurrence < 0:
            occurrence = len(all_occurrences) + occurrence

        if occurrence < 0 or occurrence >= len(all_occurrences):
            raise ValueError(
                f"Occurrence {occurrence} out of range. "
                f"Pattern '{node_pattern}' appears {len(all_occurrences)} times."
            )

        positions = all_occurrences[occurrence]
        # Get embeddings for all tokens in this occurrence and average
        node_embeddings = embeddings[layer, positions, :]
        return node_embeddings.mean(dim=0)

    def get_node_embedding_by_occurrence(
        self,
        embeddings: torch.Tensor,
        token_info: list[TokenInfo],
        proxy_token: str,
        occurrence: int,
        layer: int,
    ) -> torch.Tensor:
        """
        Get embedding of a node using a proxy token at a specific occurrence.

        Args:
            embeddings: Full embedding tensor
            token_info: Token information
            proxy_token: The token to use as proxy (e.g., '1' for '<e1>')
            occurrence: Which occurrence (-1 for last, 0 for first, etc.)
            layer: Which layer to extract from

        Returns:
            Embedding vector of shape (d_model,)
        """
        positions = self.find_all_occurrences(token_info, proxy_token)

        if not positions:
            raise ValueError(f"Token '{proxy_token}' not found in sequence")

        # Handle negative indexing (e.g., -1 for last)
        if occurrence < 0:
            occurrence = len(positions) + occurrence

        if occurrence < 0 or occurrence >= len(positions):
            raise ValueError(
                f"Occurrence {occurrence} out of range. "
                f"Token '{proxy_token}' appears {len(positions)} times."
            )

        pos = positions[occurrence]
        return embeddings[layer, pos, :]


def load_sample(sample_path: str | Path) -> dict:
    """Load a sample JSON file."""
    with open(sample_path, "r") as f:
        return json.load(f)


def print_tokenization(token_info: list[TokenInfo]):
    """Pretty print tokenization information."""
    print("\nTokenization:")
    print("-" * 50)
    for info in token_info:
        print(f"  Position {info.position:3d}: '{info.token_str}' (id={info.token_id})")
    print("-" * 50)
