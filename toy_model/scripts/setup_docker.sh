#!/bin/bash
# Docker environment setup script
# Usage: ./setup_docker.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Emergent Analogy - Docker Setup ==="
echo "Project directory: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# WandB Configuration
WANDB_API_KEY=

# GPU Configuration (comma-separated GPU IDs, e.g., "0,1" or "0")
CUDA_VISIBLE_DEVICES=0

# WandB Project Settings (can be overridden at runtime)
WANDB_PROJECT=emergent_analogy
WANDB_RUN_NAME=
EOF
    echo ".env file created. Please edit it to add your WANDB_API_KEY."
else
    echo ".env file already exists."
fi

# Build Docker image
echo ""
echo "Building Docker image..."
docker compose build

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env file to set WANDB_API_KEY (optional)"
echo "  2. Run: ./scripts/run_docker.sh --help"
echo ""
