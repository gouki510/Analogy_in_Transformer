#!/bin/bash
#
# Docker run script for analogy_dirichlet analysis
#
# Usage:
#   ./run_docker.sh                           # Run with default settings (sample_1.json, CUDA)
#   ./run_docker.sh --sample samples/sample_2.json
#   ./run_docker.sh --device cpu              # Run on CPU
#   ./run_docker.sh --no_rmsnorm --proxy_method mean
#
# Environment variables:
#   USE_GPU=0       # Set to 0 to disable GPU
#   HF_TOKEN=xxx    # Hugging Face token (for gated models)
#

set -e

# Default values
IMAGE_NAME="analogy-dirichlet"
CONTAINER_NAME="analogy-dirichlet-run"

# Check if GPU should be used
USE_GPU="${USE_GPU:-1}"
GPU_ID="${GPU_ID:-}"  # Specify GPU ID (e.g., GPU_ID=1)
GPU_FLAG=""
DEVICE="cuda"

if [ "$USE_GPU" = "1" ] && command -v nvidia-smi &> /dev/null; then
    if [ -n "$GPU_ID" ]; then
        GPU_FLAG="--gpus device=$GPU_ID"
        echo "Using GPU $GPU_ID"
    else
        GPU_FLAG="--gpus all"
        echo "Using all GPUs"
    fi
else
    GPU_FLAG=""
    DEVICE="cpu"
    echo "Running on CPU"
fi

# Build the image if it doesn't exist or if --build is passed
if [ "$1" = "--build" ]; then
    shift
    echo "Building Docker image..."
    docker build -t "$IMAGE_NAME" .
elif ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "Building Docker image (first run)..."
    docker build -t "$IMAGE_NAME" .
fi

# Create results directory if it doesn't exist
mkdir -p results

# Hugging Face token (default or from environment)
HF_TOKEN="${HF_TOKEN:-}"  # Set your HuggingFace token
HF_ENV="-e HF_TOKEN=$HF_TOKEN"

# Run the container
echo "Running analysis..."
docker run --rm \
    $GPU_FLAG \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/samples:/app/samples" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    $HF_ENV \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" \
    --device "$DEVICE" \
    --plot \
    "$@"

echo "Analysis complete. Results saved to ./results/"
