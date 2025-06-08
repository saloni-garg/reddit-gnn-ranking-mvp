#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(docker images -q pyg-m2:latest 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build -t pyg-m2:latest "$PROJECT_ROOT/docker"
fi

docker run --rm \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    -e PYTHONPATH=/workspace \
    pyg-m2:latest \
    python src/training/train.py 