# Toy Model: Synthetic Task Experiments

This module implements the synthetic task for studying emergent analogical reasoning in Transformers (Sections 2-4 of the paper).

## Overview

The experiment trains a GPT-2-like model on synthetic knowledge graph data with three types of facts:

- **Atomic facts**: Direct entity-relation-entity triples `(es, r, et)`
- **Compositional facts**: 2-hop relational compositions `(es, r1, r2, et)`  
- **Analogical facts**: Functor mappings between entity categories `(es, f, F(es))`

The model learns to generalize across different distribution types:

- **ID (In-Distribution)**: Facts seen during training
- **Near-OOD**: Novel compositions where one hop is in-distribution
- **Far-OOD**: Compositions where both hops are out-of-distribution
- **Analogical OOD**: Functor mappings for entity pairs not seen during training

## Project Structure

```
toy_model/
├── configs/                  # Configuration files
│   └── default.yaml
├── src/                      # Source code
│   ├── data/                 # Dataset generation
│   │   ├── builder.py
│   │   └── dataset.py
│   ├── model/                # Model architecture
│   │   └── gpt2.py           # GPT-2 with RoPE
│   ├── generate_data.py
│   └── train.py
├── notebooks/                # Analysis notebooks
│   ├── training_dynamics.ipynb
│   └── dirichlet_energy_analysis.ipynb
├── scripts/                  # Docker and run scripts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### Option 1: Docker (Recommended)

```bash
cd toy_model

# Setup Docker environment (build image + create .env file)
./scripts/setup_docker.sh

# (Optional) Edit .env file to set WandB API key
vi .env
# Or copy from example:
cp env.example .env

# Run data generation + training
./scripts/run_docker.sh

# Data generation only
./scripts/run_docker.sh --mode data

# Training only
./scripts/run_docker.sh --mode train
```

#### Advanced Docker Options

```bash
# Enable WandB logging
./scripts/run_docker.sh --wandb-key YOUR_API_KEY --wandb-project my_project --wandb-run experiment_1

# Custom training parameters
./scripts/run_docker.sh --epochs 500 --batch-size 32 --lr 0.001

# Run on CPU
./scripts/run_docker.sh --gpu -1

# Disable WandB
./scripts/run_docker.sh --no-wandb

# Interactive shell
./scripts/run_docker.sh --interactive

# Show help
./scripts/run_docker.sh --help
```

#### Direct Docker Compose Commands

```bash
# Generate data
docker compose run --rm generate-data

# Train with GPU
docker compose run --rm train

# Train with CPU
docker compose run --rm train-cpu

# Interactive shell
docker compose run --rm emergent-analogy bash
```

### Option 2: Local Installation

#### Prerequisites
- Python 3.10+
- CUDA 12.1+ (for GPU support)

#### Installation

```bash
cd toy_model
pip install -r requirements.txt
```

#### Generate Data

```bash
cd src
python generate_data.py --config ../configs/default.yaml
```

#### Train

```bash
cd src
python train.py --config ../configs/default.yaml
```

## Configuration

Edit `configs/default.yaml` to customize experiment settings:

### Data Settings
```yaml
data:
  num_entities: 10        # Total number of entities |E|
  num_relations: 10000    # Number of relations |R|
  sub_size: 5             # Size of E1 and E2 subsets
  atomic_ood_ratio: 0.0   # Ratio of OOD atomic facts
  compositional_ood_ratio: 0.1
  analogical_ood_ratio: 0.4
```

### Model Settings
```yaml
d_model: 128    # Model dimension
n_layer: 1      # Number of transformer layers (paper default: 1)
n_head: 1       # Number of attention heads
max_len: 64     # Maximum sequence length
```

### Training Settings
```yaml
batch_size: 64
epochs: 1000
lr: 1e-4
weight_decay: 0.0
```

## Command Line Options

### generate_data.py
```bash
python generate_data.py \
    --config ../configs/default.yaml \
    --output_dir ../data/custom_dataset \
    --num_entities 20 \
    --num_relations 5000 \
    --seed 123
```

### train.py
```bash
python train.py \
    --config ../configs/default.yaml \
    --data_dir ../data/custom_dataset \
    --save_dir ../runs/custom_run \
    --epochs 500 \
    --batch_size 32 \
    --lr 0.001 \
    --no_wandb
```

## W&B Logging

### Method 1: Using run_docker.sh (Recommended)

```bash
./scripts/run_docker.sh \
    --wandb-key YOUR_API_KEY \
    --wandb-project my_project \
    --wandb-run experiment_name
```

### Method 2: Using .env file

```bash
echo "WANDB_API_KEY=your_api_key" >> .env
echo "WANDB_PROJECT=my_project" >> .env
echo "WANDB_RUN_NAME=experiment_1" >> .env

./scripts/run_docker.sh
```

### Method 3: Environment Variables

```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=my_project
export WANDB_RUN_NAME=experiment_1

./scripts/run_docker.sh
```

### Method 4: Direct train.py arguments

```bash
python train.py \
    --config ../configs/default.yaml \
    --wandb_project my_project \
    --wandb_run experiment_name
```

### Priority Order
Environment variables > Command line arguments > config.yaml > Default values

## Evaluation Metrics

The model is evaluated on the following metrics:

- **CE (Cross Entropy)**: Lower is better
- **PPL (Perplexity)**: exp(CE), lower is better
- **ACC (Accuracy)**: Exact match accuracy for the target entity
- **PROB**: Probability assigned to the correct target token

Results are reported separately for each data type:

| Type | Description |
|------|-------------|
| `id_atomic` | In-distribution atomic facts |
| `ood_atomic` | Out-of-distribution atomic facts |
| `id_compositional` | In-distribution compositional facts |
| `near_ood_compositional` | Near-OOD compositional facts |
| `far_ood_compositional` | Far-OOD compositional facts |
| `id_analogical` | In-distribution functor mappings |
| `ood_analogical` | Out-of-distribution functor mappings |

## License

MIT License
