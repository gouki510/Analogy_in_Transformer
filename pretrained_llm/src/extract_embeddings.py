"""
Extract and save embeddings for inspection.

Saves embeddings of relevant tokens at all layers for manual verification.
"""

import torch
import json
from pathlib import Path
from transformer_lens import HookedTransformer


def main():
    print("Loading model...")
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b", device="cpu", dtype=torch.float32
    )

    prompt = "<e1>a<e2>, <e1>b<e3>. <e6>a<e4>, <e6>b<e7>. <e1>~<e6>, <e3>~<e"

    # Tokenize
    tokens = model.to_tokens(prompt, prepend_bos=True)
    token_strs = model.to_str_tokens(prompt, prepend_bos=True)

    print(f"Prompt: {repr(prompt)}")
    print(f"Number of tokens: {len(token_strs)}")

    # Find positions for relevant tokens
    positions = {
        "<e1>": None,  # last '1'
        "<e6>": None,  # last '6'
        "<e3>": None,  # last '3'
        "<e7>": None,  # first '7'
    }

    digit_positions = {}
    for digit in ["1", "3", "6", "7"]:
        digit_positions[digit] = [i for i, t in enumerate(token_strs) if t == digit]
        print(f"Token '{digit}' at positions: {digit_positions[digit]}")

    positions["<e1>"] = digit_positions["1"][-1]  # last '1'
    positions["<e6>"] = digit_positions["6"][-1]  # last '6'
    positions["<e3>"] = digit_positions["3"][-1]  # last '3'
    positions["<e7>"] = digit_positions["7"][0]   # first '7'

    print(f"\nSelected positions:")
    for node, pos in positions.items():
        print(f"  {node} -> position {pos} (token: {repr(token_strs[pos])})")

    # Extract embeddings
    print("\nExtracting embeddings...")
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda n: "resid_post" in n or n == "hook_embed"
    )

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    print(f"\nModel config: n_layers={n_layers}, d_model={d_model}")

    # Prepare output directory
    output_dir = Path("results/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings for each node at each layer
    all_embeddings = {}

    for node, pos in positions.items():
        all_embeddings[node] = {}

        for layer in range(n_layers + 1):
            if layer == 0:
                layer_emb = cache["hook_embed"][0]
                layer_name = "embed"
            else:
                layer_emb = cache[f"blocks.{layer-1}.hook_resid_post"][0]
                layer_name = f"layer_{layer-1}"

            emb = layer_emb[pos]  # (d_model,)
            norm = emb.norm().item()

            all_embeddings[node][layer] = {
                "norm": norm,
                "mean": emb.mean().item(),
                "std": emb.std().item(),
                "min": emb.min().item(),
                "max": emb.max().item(),
            }

    # Print summary
    print("\n" + "=" * 80)
    print("EMBEDDING NORMS BY LAYER")
    print("=" * 80)
    print(f"{'Layer':<8} {'||<e1>||':<12} {'||<e6>||':<12} {'||<e3>||':<12} {'||<e7>||':<12}")
    print("-" * 56)

    for layer in range(n_layers + 1):
        norms = [all_embeddings[node][layer]["norm"] for node in ["<e1>", "<e6>", "<e3>", "<e7>"]]
        layer_name = "embed" if layer == 0 else f"{layer-1}"
        print(f"{layer_name:<8} {norms[0]:<12.2f} {norms[1]:<12.2f} {norms[2]:<12.2f} {norms[3]:<12.2f}")

    # Save raw embeddings as tensors
    print("\n" + "=" * 80)
    print("SAVING RAW EMBEDDINGS")
    print("=" * 80)

    embeddings_to_save = {}

    for node, pos in positions.items():
        embeddings_to_save[node] = {}

        for layer in range(n_layers + 1):
            if layer == 0:
                layer_emb = cache["hook_embed"][0]
            else:
                layer_emb = cache[f"blocks.{layer-1}.hook_resid_post"][0]

            emb = layer_emb[pos].detach().cpu()
            embeddings_to_save[node][layer] = emb

    # Save as a single file
    torch.save(embeddings_to_save, output_dir / "node_embeddings.pt")
    print(f"Saved embeddings to {output_dir / 'node_embeddings.pt'}")

    # Also save statistics as JSON
    with open(output_dir / "embedding_stats.json", "w") as f:
        json.dump(all_embeddings, f, indent=2)
    print(f"Saved statistics to {output_dir / 'embedding_stats.json'}")

    # Save full residual stream for last position (for logit lens verification)
    last_pos = len(token_strs) - 1
    last_pos_embeddings = {}

    for layer in range(n_layers + 1):
        if layer == 0:
            layer_emb = cache["hook_embed"][0]
        else:
            layer_emb = cache[f"blocks.{layer-1}.hook_resid_post"][0]

        last_pos_embeddings[layer] = layer_emb[last_pos].detach().cpu()

    torch.save(last_pos_embeddings, output_dir / "last_position_embeddings.pt")
    print(f"Saved last position embeddings to {output_dir / 'last_position_embeddings.pt'}")

    # Print some specific layer details for verification
    print("\n" + "=" * 80)
    print("DETAILED STATS FOR SELECTED LAYERS")
    print("=" * 80)

    for layer in [0, 13, 26]:
        print(f"\nLayer {layer}:")
        for node in ["<e1>", "<e6>", "<e3>", "<e7>"]:
            stats = all_embeddings[node][layer]
            print(f"  {node}: norm={stats['norm']:.4f}, mean={stats['mean']:.6f}, "
                  f"std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")


if __name__ == "__main__":
    main()
