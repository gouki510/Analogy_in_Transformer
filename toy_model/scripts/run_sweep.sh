#!/bin/bash
# Script to run sweep experiments
# Usage: ./run_sweep.sh [OPTIONS]
#
# Options:
#   --mode <grid|random>        Sweep mode (default: grid)
#   --n_samples <num>           Number of samples for random search (default: 10)
#   --sweep_data                Sweep data parameters
#   --sweep_optimizer           Sweep optimizer parameters
#   --wandb-key <key>           WandB API Key
#   --wandb-project <name>      WandB project name
#   --dry-run                   Show plan without executing
#   --gpu <id>                  GPU ID to use (default: 0)
#   --help                      Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

#############################################
# Edit here to configure sweep targets
#############################################
# Specify true/false
DO_SWEEP_DATA=false
DO_SWEEP_OPTIMIZER=true

# WandB settings
WANDB_KEY=""

#############################################
# Other settings
#############################################
MODE="grid"
N_SAMPLES=10
DRY_RUN=""
GPU_ID="0"
CONFIG="configs/sweep_config.yaml"

# Convert sweep targets to flags
SWEEP_DATA=""
SWEEP_OPTIMIZER=""
if [ "$DO_SWEEP_DATA" = true ]; then
    SWEEP_DATA="--sweep_data"
fi
if [ "$DO_SWEEP_OPTIMIZER" = true ]; then
    SWEEP_OPTIMIZER="--sweep_optimizer"
fi

# Load .env file if exists
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --sweep_data)
            SWEEP_DATA="--sweep_data"
            shift
            ;;
        --sweep_optimizer)
            SWEEP_OPTIMIZER="--sweep_optimizer"
            shift
            ;;
        --wandb-key)
            WANDB_KEY="$2"
            shift 2
            ;;
        --wandb-project)
            # Override from command line (not usually needed)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode <grid|random>        Sweep mode (default: grid)"
            echo "  --n_samples <num>           Number of samples for random search (default: 10)"
            echo "  --sweep_data                Sweep data parameters"
            echo "  --sweep_optimizer           Sweep optimizer parameters"
            echo "  --wandb-key <key>           WandB API Key"
            echo "  --wandb-project <name>      WandB project name (default: emergent_analogy_sweep)"
            echo "  --dry-run                   Show plan without executing"
            echo "  --gpu <id>                  GPU ID to use (default: 0)"
            echo "  --config <path>             Sweep config file (default: configs/sweep_config.yaml)"
            echo "  --help                      Show help"
            echo ""
            echo "Examples:"
            echo "  # Grid search on data parameters only"
            echo "  $0 --sweep_data --wandb-project my_data_sweep"
            echo ""
            echo "  # Random search on optimizer parameters (20 samples)"
            echo "  $0 --sweep_optimizer --mode random --n_samples 20"
            echo ""
            echo "  # Sweep both (default)"
            echo "  $0 --wandb-project full_sweep"
            echo ""
            echo "  # Dry run (show plan only)"
            echo "  $0 --dry-run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

# Auto-set WandB Project based on sweep type
DATE_STR=$(date +%Y%m%d)
if [ -n "$SWEEP_DATA" ] && [ -z "$SWEEP_OPTIMIZER" ]; then
    # Data parameters only
    WANDB_PROJECT="EA_data_${DATE_STR}"
elif [ -z "$SWEEP_DATA" ] && [ -n "$SWEEP_OPTIMIZER" ]; then
    # Optimizer parameters only
    WANDB_PROJECT="EA_optim_${DATE_STR}"
else
    # Both
    WANDB_PROJECT="EA_full_${DATE_STR}"
fi

echo "=== Emergent Analogy - Sweep Experiments ==="
echo "Mode: $MODE"
echo "Sweep Type: ${SWEEP_DATA:-data} ${SWEEP_OPTIMIZER:-optimizer}"
echo "WandB Project: $WANDB_PROJECT"
echo "GPU: $GPU_ID"
echo ""

# Set environment variables
ENV_VARS="-e CUDA_VISIBLE_DEVICES=$GPU_ID"

# WandB settings
if [ -n "$WANDB_KEY" ]; then
    ENV_VARS="$ENV_VARS -e WANDB_API_KEY=$WANDB_KEY"
elif [ -n "$WANDB_API_KEY" ]; then
    ENV_VARS="$ENV_VARS -e WANDB_API_KEY=$WANDB_API_KEY"
fi

ENV_VARS="$ENV_VARS -e WANDB_PROJECT=$WANDB_PROJECT"

# Build sweep script arguments
SWEEP_ARGS="--config /app/$CONFIG --mode $MODE --n_samples $N_SAMPLES --wandb_project $WANDB_PROJECT"

if [ -n "$SWEEP_DATA" ]; then
    SWEEP_ARGS="$SWEEP_ARGS $SWEEP_DATA"
fi

if [ -n "$SWEEP_OPTIMIZER" ]; then
    SWEEP_ARGS="$SWEEP_ARGS $SWEEP_OPTIMIZER"
fi

if [ -n "$DRY_RUN" ]; then
    SWEEP_ARGS="$SWEEP_ARGS $DRY_RUN"
fi

echo "Running sweep..."
docker compose run --rm $ENV_VARS emergent-analogy \
    python sweep.py $SWEEP_ARGS

echo ""
echo "=== Sweep Complete ==="
