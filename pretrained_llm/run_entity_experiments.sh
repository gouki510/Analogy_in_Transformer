#!/bin/bash
#
# Run experiments for multiple entity counts (3, 4, 5, 7)
#
# Usage:
#   ./run_entity_experiments.sh                                    # Run with default model (gemma-2-2b)
#   ./run_entity_experiments.sh --model meta-llama/Llama-3.1-8B    # Llama
#   ./run_entity_experiments.sh --model google/gemma-2-9b          # Gemma 9B
#   ./run_entity_experiments.sh --output results/my-experiment     # Custom output directory
#

set -e

# Default model (Gemma 2 2B)
MODEL="${MODEL:-google/gemma-2-2b}"
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-generate output directory from model name if not specified
if [ -z "$OUTPUT_DIR" ]; then
    # Extract model short name (e.g., meta-llama/Llama-3.1-8B -> llama-3.1-8b)
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
    OUTPUT_DIR="results/$MODEL_SHORT"
fi

echo "=============================================="
echo "Running entity count experiments"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR"
echo "=============================================="

# Sample files for each entity count
SAMPLES=(
    "samples/sample_1.json"           # 3 entities
    "samples/sample_4entities.json"   # 4 entities  
    "samples/sample_5entities.json"   # 5 entities
    "samples/sample_7entities.json"   # 7 entities
)

ENTITY_COUNTS=(3 4 5 7)

# Run experiments
for i in "${!SAMPLES[@]}"; do
    sample="${SAMPLES[$i]}"
    ent="${ENTITY_COUNTS[$i]}"
    
    echo ""
    echo "=============================================="
    echo "Running ${ent}-entity experiment: $sample"
    echo "=============================================="
    
    ./run_docker.sh --sample "$sample" --model "$MODEL" --output "$OUTPUT_DIR"
    
    echo "Completed ${ent}-entity experiment"
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
