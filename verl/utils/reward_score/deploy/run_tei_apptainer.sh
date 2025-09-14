#!/bin/bash
# TEI Server using Apptainer (Container-based approach)
# This runs the official Text Embeddings Inference server in a container

set -e

# Configuration
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}"
export PORT="${PORT:-8080}"
export MAX_BATCH_TOKENS="${MAX_BATCH_TOKENS:-65536}"
export MAX_BATCH_REQUESTS="${MAX_BATCH_REQUESTS:-256}"
export DTYPE="${DTYPE:-float16}"

# Create data directory for model cache
mkdir -p ~/tei-data

echo "TEI Server (Apptainer) - Starting..."
echo "======================================="
echo "Model: $MODEL_ID"
echo "Port: $PORT"
echo "Data directory: ~/tei-data"
echo "Node: $(hostname)"
echo "IP addresses: $(hostname -I)"
echo ""

# Check if Apptainer image exists
if [ ! -f "text-embeddings-inference_hopper-1.8.sif" ]; then
    echo "Error: Apptainer image not found!"
    echo "Please run: apptainer pull docker://ghcr.io/huggingface/text-embeddings-inference:hopper-1.8"
    exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. You may need it for some models."
fi

# Run the TEI server using Apptainer
echo "Starting TEI server container..."
apptainer run --nv \
  --bind ~/tei-data:/data \
  --env HF_TOKEN="$HF_TOKEN" \
  text-embeddings-inference_hopper-1.8.sif \
  --model-id $MODEL_ID \
  --max-batch-tokens $MAX_BATCH_TOKENS \
  --max-batch-requests $MAX_BATCH_REQUESTS \
  --dtype $DTYPE \
  --port $PORT
