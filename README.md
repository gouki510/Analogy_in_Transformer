# Emergent Analogical Reasoning in Transformers

This repository provides the official implementation for the paper:

**"Emergent Analogical Reasoning in Transformers"**
*Gouki Minegishi, Jingyuan Feng, Hiroki Furuta, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo*

## Abstract

Analogy is a central faculty of human intelligence, enabling abstract patterns discovered in one domain to be applied to another. Despite its central role in cognition, the mechanisms by which Transformers acquire and implement analogical reasoning remain poorly understood. In this work, inspired by the notion of functors in category theory, we formalize analogical reasoning as the inference of correspondences between entities across categories. Based on this formulation, we introduce synthetic tasks that evaluate the emergence of analogical reasoning under controlled settings. We find that the emergence of analogical reasoning is highly sensitive to data characteristics, optimization choices, and model scale. Through mechanistic analysis, we show that analogical reasoning in Transformers decomposes into two key components: (1) geometric alignment of relational structure in the embedding space, and (2) the application of a functor within the Transformer. These mechanisms enable models to transfer relational structure from one category to another, realizing analogy. Finally, we quantify these effects and find that the same trends are observed in pretrained LLMs.

## Repository Structure

```
.
├── toy_model/                 # Synthetic task experiments (Section 2-4)
│   ├── src/                   # Source code (model, data, training)
│   ├── configs/               # Configuration files
│   ├── scripts/               # Docker and run scripts
│   └── notebooks/             # Analysis notebooks
│
├── pretrained_llm/            # Pretrained LLM experiments (Section 5)
│   ├── src/                   # Source code (embeddings, metrics)
│   ├── samples/               # Sample JSON configurations
│   ├── notebooks/             # Analysis notebooks
│   └── main.py                # Main analysis script
│
└── README.md                  # This file
```

## Experiments

### 1. Toy Model Experiments (`toy_model/`)

This module implements the synthetic task for studying analogical reasoning in Transformers (Sections 2-4 of the paper).

**Key Features:**

- Synthetic knowledge graph with atomic, compositional, and analogical facts
- GPT-2-like Transformer with Rotary Position Embedding (RoPE)
- Training dynamics analysis for compositional and analogical reasoning
- Mechanistic analysis: Dirichlet Energy, attention patterns, parallelism

See [`toy_model/README.md`](toy_model/README.md) for detailed instructions.

### 2. Pretrained LLM Experiments (`pretrained_llm/`)

This module analyzes analogical reasoning mechanisms in pretrained LLMs using in-context learning (Section 5 of the paper).

**Key Features:**

- Layer-wise Dirichlet Energy analysis
- Logit lens for tracking prediction evolution
- PCA visualization of hidden states
- Support for Gemma-2 and LLaMA models

See [`pretrained_llm/README.md`](pretrained_llm/README.md) for detailed instructions.

## Quick Start

### Toy Model (Synthetic Task)

```bash
cd toy_model

# Install dependencies
pip install -r requirements.txt

# Generate data
cd src && python generate_data.py --config ../configs/default.yaml

# Train model
python train.py --config ../configs/default.yaml
```

### Pretrained LLM Analysis

```bash
cd pretrained_llm

# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py --sample samples/sample_1.json --device cuda --plot
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA 12.1+ (for GPU support)

See individual `requirements.txt` files for complete dependencies.
# Analogy_in_Transformer
