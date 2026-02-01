# Pretrained LLM: Analogy Analysis

This module analyzes how pretrained LLMs process analogical reasoning by computing Dirichlet energy across layers (Section 5 of the paper).

## Task

Given the prompt:

```
<e1>a<e2>, <e1>b<e3>. <e6>a<e4>, <e6>b<e7>. <e1>~<e6>, <e3>~<e
```

The model predicts `7`, completing `<e3>~<e7>`. The `~` symbol represents the functor (analogy mapping).

Two categories share the same relational structure:
- **Category 1**: `<e1>`, `<e2>`, `<e3>` with relations `a`, `b`
- **Category 2**: `<e6>`, `<e4>`, `<e7>` with relations `a`, `b`

## Core Hypothesis

As the model processes the analogy through layers, embeddings of functorially-related entities (`<e1>` and `<e6>`) should become more similar in deeper layers, correlating with correct prediction.

## Project Structure

```
pretrained_llm/
├── src/                      # Source code
│   ├── embeddings.py         # Embedding extraction with RMSNorm
│   ├── dirichlet_energy.py   # Dirichlet energy computation
│   ├── logit_lens.py         # Logit lens analysis
│   └── functor_similarity.py # Functor similarity metrics
├── notebooks/                # Analysis notebooks
│   ├── analysis.ipynb
│   └── pca_visualization.ipynb
├── samples/                  # Sample JSON configurations
│   ├── sample_1.json
│   ├── sample_2.json
│   └── sample_4_entities.json
├── main.py                   # Main analysis script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Metrics

### Dirichlet Energy

$$E(G) = \sum_{(i,j) \in \text{edges}} \|x_i - x_j\|^2$$

- **Lower energy** = more similar embeddings between functor-related entities
- Expected: Energy **decreases** as P(target) **increases**

### Normalized Dirichlet Energy

$$E_{norm} = \frac{E(G)}{\text{mean}_{t}(\|x_t\|^2)}$$

Normalizes by average squared norm of all token embeddings at each layer.

## Usage

### Option 1: Docker (Recommended)

```bash
cd pretrained_llm

# Run with shell script
./run_docker.sh --sample samples/sample_1.json

# Run on CPU
USE_GPU=0 ./run_docker.sh --sample samples/sample_1.json

# Additional options
./run_docker.sh --sample samples/sample_1.json --no_rmsnorm --proxy_method mean
```

#### Docker Compose

```bash
# With GPU
docker compose up analogy-analysis

# With CPU
docker compose up analogy-analysis-cpu
```

#### Direct Docker Execution

```bash
# Build image
docker build -t analogy-dirichlet .

# Run with GPU
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/samples:/app/samples \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  analogy-dirichlet --sample samples/sample_1.json --device cuda --plot

# Run with CPU
docker run \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/samples:/app/samples \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  analogy-dirichlet --sample samples/sample_1.json --device cpu --plot
```

### Option 2: Local Installation

```bash
cd pretrained_llm

# Install dependencies
pip install -r requirements.txt

# num_proxy (default): use single token as proxy
python main.py --sample samples/sample_1.json --proxy_method position --plot --device cpu

# mean_proxy: use mean of all entity tokens
python main.py --sample samples/sample_1.json --proxy_method mean --plot --device cpu

# Without RMSNorm (raw residual stream) - for comparison
python main.py --sample samples/sample_1.json --proxy_method mean --no_rmsnorm --plot --device cpu
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample` | `samples/sample_1.json` | Sample JSON file |
| `--model` | `google/gemma-2-2b` | Model name |
| `--device` | `cpu` | Device (`cpu`, `cuda`, `mps`) |
| `--proxy_method` | `position` | `position` (num_proxy) or `mean` (mean_proxy) |
| `--no_rmsnorm` | `False` | Disable RMSNorm (use raw residual stream) |
| `--plot` | `False` | Generate plots |

## Proxy Methods

Since `<e1>` tokenizes to multiple tokens (`<`, `e`, `1`, `>`), we provide two methods:

| Method | Flag | Entity Embedding | Use Case |
|--------|------|------------------|----------|
| **num_proxy** | `--proxy_method position` | Single token (e.g., `1` for `<e1>`) | Simpler, often sufficient |
| **mean_proxy** | `--proxy_method mean` | Mean of all tokens in entity | More complete representation |

## Sample Configuration

```json
{
    "name": "abstract_analogy_1",
    "prompt": "<e1>a<e2>, <e1>b<e3>. <e6>a<e4>, <e6>b<e7>. <e1>~<e6>, <e3>~<e",
    "target_token": "7",
    "graph": {
        "edges": [{"source": "<e1>", "target": "<e6>", "type": "functor"}]
    },
    "node_positions": {
        "<e1>": {"proxy_token": "1", "occurrence": -1},
        "<e6>": {"proxy_token": "6", "occurrence": -1}
    }
}
```

**Note**: Only include **complete** functor edges. `<e3>~<e7>` is incomplete (target `7` is being predicted).

## Output Structure

Results are organized by: `results/<sample_name>/<norm_method>/<proxy_method>/`

```
results/
└── abstract_analogy_1/
    ├── rmsnorm/
    │   ├── num_proxy/
    │   │   ├── analysis_results.json
    │   │   ├── dirichlet_vs_probability.pdf
    │   │   └── correlation_plot.pdf
    │   └── mean_proxy/
    └── no_rmsnorm/
```

## Expected Results

### With RMSNorm (default)
**Negative correlation** (-0.67 to -0.80) confirms the hypothesis — as Dirichlet energy decreases (entities become more similar), prediction probability increases.

### Without RMSNorm (`--no_rmsnorm`)
**Positive correlation** (~+0.80) is observed because raw embedding norms grow across layers, confounding the Dirichlet energy measure.

## License

MIT License
