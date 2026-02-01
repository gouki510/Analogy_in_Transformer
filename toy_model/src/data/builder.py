"""
Dataset builder for compositional and functor-based knowledge graph datasets.
"""

import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def build_dicts(items):
    """Build index-to-item and item-to-index dictionaries."""
    ind2 = {i: it for i, it in enumerate(items)}
    toind = {it: i for i, it in ind2.items()}
    return ind2, toind


def split(items, n_ood, rng=None):
    """
    Split items into OOD / ID lists.
    Returns (ood_list, id_list).
    """
    if rng is None:
        rng = np.random.default_rng()
    items_list = list(items)
    rng.shuffle(items_list)
    n_ood = max(0, min(n_ood, len(items_list)))
    ood = items_list[:n_ood]
    ide = items_list[n_ood:]
    return ood, ide


def form_items(c: List[str], t: str) -> Dict[str, str]:
    """Form input/target text pairs."""
    input_text = "".join(c)
    target_text = input_text + "".join([t])
    return {
        "input_text": input_text,
        "target_text": target_text
    }


def build_dataset_with_functor(
    num_entities: int,
    num_relations: int,
    sub_size: int,
    atomic_ood_ratio: float = 0.05,
    compositional_ood_ratio: float = 0.005,
    analogical_ood_ratio: float = 0.10,
    seed: int = None,
    include_f_inverse: bool = False,
    duplicate_relation: bool = True
) -> Tuple:
    """
    Build a dataset with functor structure for analogical reasoning experiments.
    
    Args:
        num_entities: Total number of entities
        num_relations: Number of relations
        sub_size: Size of E1 and E2 subsets (|E1| = |E2|)
        atomic_ood_ratio: OOD ratio for atomic facts
        compositional_ood_ratio: OOD ratio for compositional (2-hop) facts
        analogical_ood_ratio: OOD ratio for analogical facts
        seed: Random seed
        include_f_inverse: Whether to include inverse functor <f_inv>
        duplicate_relation: Whether to allow duplicate relations
        
    Returns:
        Tuple containing entities, relations, and various fact sets
    """
    rng = np.random.default_rng(seed)

    # Build vocabulary
    entities = [f"<e_{i}>" for i in range(num_entities)]
    ind2entity, entity2ind = build_dicts(entities)

    base_relations = [f"<r_{i}>" for i in range(num_relations)]
    extra_relations = ["<f>"] + (["<f_inv>"] if include_f_inverse else [])
    relations = base_relations + extra_relations
    ind2relation, relation2ind = build_dicts(relations)
    F = "<f>"
    F_INV = "<f_inv>" if include_f_inverse else None

    assert 2 * sub_size <= num_entities, "sub_size is too large (E1 and E2 must be disjoint)"

    # Split entities into E1, E2, and OTHER
    perm = rng.permutation(num_entities)
    E1_idx = perm[:sub_size].tolist()
    E2_idx = perm[sub_size:2*sub_size].tolist()
    OTHER_idx = perm[2*sub_size:].tolist()

    E1 = [ind2entity[i] for i in E1_idx]
    E2 = [ind2entity[i] for i in E2_idx]

    # f is a bijection from E1 to E2
    E2_perm = rng.permutation(E2_idx).tolist()
    f_map = {ind2entity[i]: ind2entity[j] for i, j in zip(E1_idx, E2_perm)}
    if include_f_inverse:
        f_inv_map = {v: k for k, v in f_map.items()}

    # Relation subset for E1/E2
    rsubset_size = max(1, num_relations)
    R_sub_idx = rng.choice(num_relations, size=rsubset_size, replace=False).tolist()

    # Build atomic triples (h, r, t)
    atomic_dict = defaultdict(list)
    atomics = []

    # E1: internal edges
    for hi in E1_idx:
        used_r = set()
        for ti in E1_idx:
            if ti == hi:
                continue
            if duplicate_relation:
                r_idx = rng.choice(R_sub_idx)
            else:
                available_r = [r for r in R_sub_idx if r not in used_r]
                if not available_r:
                    break
                r_idx = rng.choice(available_r)
                used_r.add(r_idx)
            h, r, t = ind2entity[hi], ind2relation[r_idx], ind2entity[ti]
            atomics.append((h, r, t))
            atomic_dict[h].append((r, t))

    # OTHER: noise entities (not used in main experiment)
    for hi in OTHER_idx:
        for ti in OTHER_idx:
            if ti == hi:
                continue
            r_idx = rng.choice(R_sub_idx)
            h, r, t = ind2entity[hi], ind2relation[r_idx], ind2entity[ti]
            atomics.append((h, r, t))
            atomic_dict[h].append((r, t))

    # OOD split for atomic facts
    n_ood_atomic = round(len(atomics) * (1-atomic_ood_ratio))
    ID_atomic_list, OOD_atomic_list = split(atomics, n_ood_atomic, rng=rng)
    ID_atomic_facts, OOD_atomic_facts = set(ID_atomic_list), set(OOD_atomic_list)

    # Create E2 atomic facts with same relation structure as E1
    for (h1, r, t1) in ID_atomic_list:
        if h1 in f_map and t1 in f_map:
            h2, t2 = f_map[h1], f_map[t1]
            edge2 = (h2, r, t2)
            ID_atomic_facts.add(edge2)
            atomic_dict[h2].append((r, t2))

    # Convert to JSON records
    id_atomic_facts = [form_items([h, r], t) for (h, r, t) in sorted(ID_atomic_facts)]
    ood_atomic_facts = [form_items([h, r], t) for (h, r, t) in sorted(OOD_atomic_facts)]

    # Compositional Facts (2-hop inference)
    id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts = [], [], []
    for ent in entities:
        for (r1, b) in atomic_dict[ent]:
            for (r2, t) in atomic_dict[b]:
                s1 = (ent, r1, b)
                s2 = (b, r2, t)
                if ent == t:
                    continue
                if (s1 in OOD_atomic_facts) or (s2 in OOD_atomic_facts):
                    far_ood_compositional_facts.append(form_items([ent, r1, r2], t))
                else:
                    if rng.uniform() > compositional_ood_ratio:
                        id_compositional_facts.append(form_items([ent, r1, r2], t))
                    else:
                        near_ood_compositional_facts.append(form_items([ent, r1, r2], t))

    # Analogical Facts
    analogical_facts = [(e1, F, f_map[e1]) for e1 in E1]
    if include_f_inverse:
        inv_analogical_facts = [(f_map[e1], F_INV, e1) for e1 in E1]
    else:
        inv_analogical_facts = []
    all_analogical_facts = analogical_facts + inv_analogical_facts

    n_analogical_ood = int(round(len(all_analogical_facts) * (1-analogical_ood_ratio)))
    analogical_ID_list, analogical_OOD_list = split(all_analogical_facts, n_analogical_ood, rng=rng)
    analogical_ID, analogical_OOD = set(analogical_ID_list), set(analogical_OOD_list)

    id_analogical_facts = [form_items([h, f], t) for (h, f, t) in sorted(analogical_ID)]
    ood_analogical_facts = [form_items([h, f], t) for (h, f, t) in sorted(analogical_OOD)]

    return (
        entities, relations,
        id_atomic_facts, ood_atomic_facts,
        id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts,
        id_analogical_facts, ood_analogical_facts,
    )


def save_dataset(
    output_dir: str,
    entities: List[str],
    relations: List[str],
    id_atomic_facts: List[Dict],
    ood_atomic_facts: List[Dict],
    id_compositional_facts: List[Dict],
    near_ood_compositional_facts: List[Dict],
    far_ood_compositional_facts: List[Dict],
    id_analogical_facts: List[Dict],
    ood_analogical_facts: List[Dict],
):
    """Save the dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Build vocabulary
    vocab = entities + relations
    
    # Create probes (test set with type labels)
    probes = []
    for item in id_atomic_facts:
        probes.append({**deepcopy(item), "type": "id_atomic"})
    for item in ood_atomic_facts:
        probes.append({**deepcopy(item), "type": "ood_atomic"})
    for item in id_compositional_facts:
        probes.append({**deepcopy(item), "type": "id_compositional"})
    for item in near_ood_compositional_facts:
        probes.append({**deepcopy(item), "type": "near_ood_compositional"})
    for item in far_ood_compositional_facts:
        probes.append({**deepcopy(item), "type": "far_ood_compositional"})
    for item in id_analogical_facts:
        probes.append({**deepcopy(item), "type": "id_analogical"})
    for item in ood_analogical_facts:
        probes.append({**deepcopy(item), "type": "ood_analogical"})
    
    # Training data = ID facts only
    train_data = id_atomic_facts + id_compositional_facts + id_analogical_facts
    
    # Save files
    files_to_save = {
        "train.json": train_data,
        "test.json": probes,
        "vocab.json": vocab,
        "ood_atomic.json": ood_atomic_facts,
        "id_compositional.json": id_compositional_facts,
        "near_ood_compositional.json": near_ood_compositional_facts,
        "far_ood_compositional.json": far_ood_compositional_facts,
        "id_analogical.json": id_analogical_facts,
        "ood_analogical.json": ood_analogical_facts,
    }
    
    for filename, data in files_to_save.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
    
    print(f"Dataset saved to {output_dir}")
    print(f"  - vocab size: {len(vocab)}")
    print(f"  - train samples: {len(train_data)}")
    print(f"  - test samples: {len(probes)}")


if __name__ == "__main__":
    # Example usage
    NUM_ENTITY_IN = 10
    NUM_RELATION = 10000
    SUB_SIZE = NUM_ENTITY_IN // 2
    
    res = build_dataset_with_functor(
        NUM_ENTITY_IN, NUM_RELATION,
        sub_size=SUB_SIZE,
        atomic_ood_ratio=0.0,
        compositional_ood_ratio=0.1,
        analogical_ood_ratio=0.4,
        seed=42,
        include_f_inverse=False,
        duplicate_relation=False
    )
    
    (entities, relations,
     id_atomic_facts, ood_atomic_facts,
     id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts,
     id_analogical_facts, ood_analogical_facts) = res
    
    print("#entities:", len(entities))
    print("#relations:", len(relations))
    print("ID atomics:", len(id_atomic_facts))
    print("OOD atomics:", len(ood_atomic_facts))
    print("ID compositional:", len(id_compositional_facts))
    print("near OOD compositional:", len(near_ood_compositional_facts))
    print("far OOD compositional:", len(far_ood_compositional_facts))
    print("ID analogical:", len(id_analogical_facts))
    print("OOD analogical:", len(ood_analogical_facts))
