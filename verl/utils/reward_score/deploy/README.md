# Remote Embedding Server Deployment

This directory contains scripts to deploy the Text Embeddings Inference (TEI) server for the remote embedding reward function.

## Quick Start

### Option 1: Python Server (Recommended)
```bash
# Install dependencies
pip install torch torchvision torchaudio transformers accelerate huggingface-hub fastapi uvicorn

# Run server
python run_tei_python.py --model-id Qwen/Qwen3-Embedding-0.6B --port 8080
```

### Option 2: Apptainer Container
```bash
# Run server (requires .sif file in this directory)
./run_tei_apptainer.sh
```

## Multi-Node Setup

1. **Start server** on GPU node
2. **Note the IP address** from server output (e.g., 172.16.0.60)
3. **Set environment variable** on training node:
   ```bash
   export EMBEDDING_SERVER_URL="http://172.16.0.60:8080"
   ```

## Testing

```bash
# Test connectivity (from parent directory)
cd ../tests/
python test_server_connectivity.py http://YOUR_SERVER_IP:8080
```

## Configuration

Both servers support environment variables:
- `MODEL_ID` - Model to use (default: Qwen/Qwen3-Embedding-0.6B)
- `PORT` - Server port (default: 8080)  
- `HF_TOKEN` - Hugging Face token (for private models)

## Files

- `run_tei_python.py` - Python-based server (direct PyTorch)
- `run_tei_apptainer.sh` - Container-based server (official TEI)
- `*.sif` - Apptainer container images
