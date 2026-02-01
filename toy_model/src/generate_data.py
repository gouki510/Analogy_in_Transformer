"""
Script to generate dataset for training.
"""

import argparse
import os

import yaml

from data.builder import build_dataset_with_functor, save_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate dataset for Emergent Analogy")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--num_entities", type=int, default=None)
    parser.add_argument("--num_relations", type=int, default=None)
    parser.add_argument("--sub_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config.get("data", {})
    
    # Override with command line arguments
    num_entities = args.num_entities or data_config.get("num_entities", 10)
    num_relations = args.num_relations or data_config.get("num_relations", 10000)
    sub_size = args.sub_size or data_config.get("sub_size", num_entities // 2)
    seed = args.seed or data_config.get("seed", 42)
    
    atomic_ood_ratio = data_config.get("atomic_ood_ratio", 0.0)
    compositional_ood_ratio = data_config.get("compositional_ood_ratio", 0.1)
    analogical_ood_ratio = data_config.get("analogical_ood_ratio", 0.4)
    include_f_inverse = data_config.get("include_f_inverse", False)
    duplicate_relation = data_config.get("duplicate_relation", False)
    
    # Generate dataset
    print("Generating dataset...")
    print(f"  num_entities: {num_entities}")
    print(f"  num_relations: {num_relations}")
    print(f"  sub_size: {sub_size}")
    print(f"  seed: {seed}")
    
    res = build_dataset_with_functor(
        num_entities, num_relations,
        sub_size=sub_size,
        atomic_ood_ratio=atomic_ood_ratio,
        compositional_ood_ratio=compositional_ood_ratio,
        analogical_ood_ratio=analogical_ood_ratio,
        seed=seed,
        include_f_inverse=include_f_inverse,
        duplicate_relation=duplicate_relation
    )
    
    (entities, relations,
     id_atomic_facts, ood_atomic_facts,
     id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts,
     id_analogical_facts, ood_analogical_facts) = res
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = data_config.get(
            "output_dir",
            f"data/composition_functor.{num_entities}.{num_relations}.{sub_size}"
        )
    
    # Save dataset
    save_dataset(
        output_dir,
        entities, relations,
        id_atomic_facts, ood_atomic_facts,
        id_compositional_facts, near_ood_compositional_facts, far_ood_compositional_facts,
        id_analogical_facts, ood_analogical_facts,
    )
    
    # Print summary
    print("\nDataset summary:")
    print(f"  #entities: {len(entities)}")
    print(f"  #relations: {len(relations)}")
    print(f"  ID atomics: {len(id_atomic_facts)}")
    print(f"  OOD atomics: {len(ood_atomic_facts)}")
    print(f"  ID compositional: {len(id_compositional_facts)}")
    print(f"  near OOD compositional: {len(near_ood_compositional_facts)}")
    print(f"  far OOD compositional: {len(far_ood_compositional_facts)}")
    print(f"  ID analogical: {len(id_analogical_facts)}")
    print(f"  OOD analogical: {len(ood_analogical_facts)}")


if __name__ == "__main__":
    main()
