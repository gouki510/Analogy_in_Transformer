#!/bin/bash
# Script to run experiments inside Docker
# Usage: ./run_docker.sh [OPTIONS]
#
# Options:
#   --mode <data|train|all>     Execution mode (default: all)
#   --wandb-key <key>           WandB API Key
#   --wandb-project <name>      WandB project name
#   --wandb-run <name>          WandB run name
#   --config <path>             Config file path (default: configs/default.yaml)
#   --gpu <id>                  GPU ID to use (default: 0, -1 for CPU)
#   --epochs <num>              Number of epochs
#   --batch-size <num>          Batch size
#   --lr <num>                  Learning rate
#   --no-wandb                  Disable WandB
#   --interactive               Interactive mode (start bash)
#   --help                      Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODE="all"
WANDB_KEY=""
WANDB_PROJECT="emergent_analogy_$(date +%Y%m%d)"
CONFIG="configs/default.yaml"
GPU_ID="0"
EPOCHS=""
BATCH_SIZE=""
LR=""
NO_WANDB=""
INTERACTIVE=""

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
        --wandb-key)
            WANDB_KEY="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-run)
            WANDB_RUN="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --no-wandb)
            NO_WANDB="1"
            shift
            ;;
        --interactive)
            INTERACTIVE="1"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode <data|train|all>     Execution mode (default: all)"
            echo "  --wandb-key <key>           WandB API Key"
            echo "  --wandb-project <name>      WandB project name"
            echo "  --wandb-run <name>          WandB run name"
            echo "  --config <path>             Config file path (default: configs/default.yaml)"
            echo "  --gpu <id>                  GPU ID to use (default: 0, -1 for CPU)"
            echo "  --epochs <num>              Number of epochs"
            echo "  --batch-size <num>          Batch size"
            echo "  --lr <num>                  Learning rate"
            echo "  --no-wandb                  Disable WandB"
            echo "  --interactive               Interactive mode (start bash)"
            echo "  --help                      Show help"
            echo ""
            echo "Examples:"
            echo "  # Data generation only"
            echo "  $0 --mode data"
            echo ""
            echo "  # Train with WandB enabled"
            echo "  $0 --mode train --wandb-key YOUR_KEY --wandb-project my_project"
            echo ""
            echo "  # Train with custom settings"
            echo "  $0 --epochs 500 --batch-size 32 --lr 0.001"
            echo ""
            echo "  # Train on CPU"
            echo "  $0 --gpu -1"
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

echo "=== Emergent Analogy - Docker Run ==="
echo "Mode: $MODE"
echo "Config: $CONFIG"
echo "GPU: $GPU_ID"

# Set environment variables
ENV_VARS="-e CUDA_VISIBLE_DEVICES=$GPU_ID"

# WandB settings
if [ -n "$WANDB_KEY" ]; then
    ENV_VARS="$ENV_VARS -e WANDB_API_KEY=$WANDB_KEY"
elif [ -n "$WANDB_API_KEY" ]; then
    ENV_VARS="$ENV_VARS -e WANDB_API_KEY=$WANDB_API_KEY"
fi

if [ -n "$WANDB_PROJECT" ]; then
    ENV_VARS="$ENV_VARS -e WANDB_PROJECT=$WANDB_PROJECT"
fi

if [ -n "$WANDB_RUN" ]; then
    ENV_VARS="$ENV_VARS -e WANDB_RUN_NAME=$WANDB_RUN"
fi

# Build train.py arguments
TRAIN_ARGS="--config /app/$CONFIG"

if [ -n "$EPOCHS" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --epochs $EPOCHS"
fi

if [ -n "$BATCH_SIZE" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --batch_size $BATCH_SIZE"
fi

if [ -n "$LR" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --lr $LR"
fi

if [ -n "$NO_WANDB" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --no_wandb"
fi

if [ -n "$WANDB_PROJECT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --wandb_project $WANDB_PROJECT"
fi

if [ -n "$WANDB_RUN" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --wandb_run $WANDB_RUN"
fi

echo ""

# Interactive mode
if [ -n "$INTERACTIVE" ]; then
    echo "Starting interactive shell..."
    if [ "$GPU_ID" == "-1" ]; then
        docker compose run --rm $ENV_VARS emergent-analogy bash
    else
        docker compose run --rm $ENV_VARS emergent-analogy bash
    fi
    exit 0
fi

# Execute
case $MODE in
    "data")
        echo "Generating dataset..."
        docker compose run --rm $ENV_VARS emergent-analogy \
            python generate_data.py --config /app/$CONFIG
        ;;
    "train")
        echo "Training model..."
        if [ "$GPU_ID" == "-1" ]; then
            docker compose run --rm $ENV_VARS emergent-analogy \
                python train.py $TRAIN_ARGS
        else
            docker compose run --rm $ENV_VARS emergent-analogy \
                python train.py $TRAIN_ARGS
        fi
        ;;
    "all")
        echo "Step 1: Generating dataset..."
        docker compose run --rm $ENV_VARS emergent-analogy \
            python generate_data.py --config /app/$CONFIG
        echo ""
        echo "Step 2: Training model..."
        if [ "$GPU_ID" == "-1" ]; then
            docker compose run --rm $ENV_VARS emergent-analogy \
                python train.py $TRAIN_ARGS
        else
            docker compose run --rm $ENV_VARS emergent-analogy \
                python train.py $TRAIN_ARGS
        fi
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: data, train, all"
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
