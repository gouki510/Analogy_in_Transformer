"""
PyTorch Dataset classes for knowledge graph training.
"""

import json
import re
from typing import Dict, List

import torch
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(r"<e_\d+>|<r_\d+>|<f>|<f_inv>")


def tokenize_strict(s: str) -> List[str]:
    """Tokenize string into entity/relation tokens."""
    toks = TOKEN_PATTERN.findall(s)
    if "".join(toks) != s:
        bad = s.replace("".join(toks), "")
        raise ValueError(f"Non-token residue: '{bad}' in '{s}'")
    return toks


class CompDataset(Dataset):
    """Dataset for compositional knowledge graph training."""
    
    def __init__(
        self,
        path_json: str,
        vocab_path: str,
        max_len: int,
        expect_type: bool = False
    ):
        """
        Args:
            path_json: Path to the JSON data file
            vocab_path: Path to the vocabulary JSON file
            max_len: Maximum sequence length
            expect_type: Whether to expect type labels in the data
        """
        with open(path_json, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        self.id2tok = {i: t for i, t in enumerate(self.vocab)}
        self.max_len = max_len
        self.expect_type = expect_type
    
    def __len__(self) -> int:
        return len(self.items)
    
    def encode(self, toks: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.tok2id[t] for t in toks]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Convert indices to tokens."""
        return [self.id2tok[i] for i in ids]
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        inp = tokenize_strict(item["input_text"])
        tgt = tokenize_strict(item["target_text"])
        
        input_ids = self.encode(tgt[:-1])
        target_ids = self.encode(tgt[1:])
        last_pos = len(target_ids) - 1
        
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "last_pos": last_pos,
            "length": len(input_ids),
        }
        
        if self.expect_type:
            out["type"] = item.get("type", "unknown")
        
        return out


def collate_pad(batch: List[Dict], pad_id: int = 0) -> Dict:
    """
    Collate function with padding.
    
    Args:
        batch: List of samples from CompDataset
        pad_id: ID to use for padding
        
    Returns:
        Batched and padded tensors
    """
    B = len(batch)
    maxL = max(ex["length"] for ex in batch)
    
    input_ids = torch.full((B, maxL), pad_id, dtype=torch.long)
    target_ids = torch.full((B, maxL), -100, dtype=torch.long)
    loss_mask = torch.zeros((B, maxL), dtype=torch.bool)
    pad_mask = torch.ones((B, maxL), dtype=torch.bool)
    types = []
    
    for i, ex in enumerate(batch):
        L = ex["length"]
        input_ids[i, :L] = ex["input_ids"]
        target_ids[i, :L] = ex["target_ids"]
        loss_mask[i, ex["last_pos"]] = True
        pad_mask[i, :L] = False
        if "type" in ex:
            types.append(ex["type"])
    
    out = {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "pad_mask": pad_mask
    }
    
    if types:
        out["type"] = types
    
    return out
