#!/bin/bash
# Run script for Emergent Analogy experiment

set -e

# Default values
CONFIG=${CONFIG:-"../configs/default.yaml"}
MODE=${MODE:-"all"}

echo "=== Emergent Analogy Experiment ==="
echo "Config: $CONFIG"
echo "Mode: $MODE"
echo ""

cd "$(dirname "$0")/../src"

case $MODE in
    "data")
        echo "Generating dataset..."
        python generate_data.py --config "$CONFIG"
        ;;
    "train")
        echo "Training model..."
        python train.py --config "$CONFIG"
        ;;
    "all")
        echo "Step 1: Generating dataset..."
        python generate_data.py --config "$CONFIG"
        echo ""
        echo "Step 2: Training model..."
        python train.py --config "$CONFIG"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: MODE=<data|train|all> ./run.sh"
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
