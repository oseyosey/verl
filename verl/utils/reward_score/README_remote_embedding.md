# Remote Embedding-based Reward Function

This module implements a remote embedding-based reward function that connects to a Text Embeddings Inference (TEI) server hosting Qwen3-Embedding-8B. This design allows GPU-accelerated embeddings without requiring GPU allocation on the reward worker.

## Overview

The remote embedding reward function (`embedding_remote.py`) mirrors the functionality of the local embedding reward (`embedding.py`) but delegates embedding computation to a remote server. This architecture is ideal for:

- **Resource Efficiency**: No GPU required on reward workers
- **Scalability**: Centralized embedding service can serve multiple training runs
- **Performance**: GPU-accelerated embeddings with Qwen3-Embedding-8B (4096-dim)
- **Flexibility**: Easy to switch between different embedding models

## Quick Start

### 1. Deploy the TEI Server

#### Using Python Server (Recommended):
```bash
cd deploy/
python run_tei_python.py --model-id Qwen/Qwen3-Embedding-0.6B --port 8080
```

#### Using Apptainer Container:
```bash
cd deploy/
./run_tei_apptainer.sh
```

### 2. Configure VERL

Set the embedding server URL:
```bash
# For same-node deployment
export EMBEDDING_SERVER_URL="http://localhost:8080"

# For multi-node deployment (use actual IP from server output)
export EMBEDDING_SERVER_URL="http://172.16.0.60:8080"
```

### 3. Use in VERL Training

In your VERL configuration, set the data source to use remote embeddings:
```yaml
data_source: embedding_remote_your_dataset
```

Or configure a custom reward function:
```yaml
custom_reward_function:
  path: verl/utils/reward_score/embedding_remote.py
  name: compute_score
```

## Architecture

```
┌─────────────────────┐         HTTP/REST API      ┌──────────────────────┐
│   VERL Training     │ ─────────────────────────> │  TEI Server          │
│   (Reward Worker)   │ <───────────────────────── │  Qwen3-Embedding-8B  │
│   - No GPU needed   │      Embeddings (4096-dim) │  - GPU accelerated   │
└─────────────────────┘                            └──────────────────────┘
```

## Features

### 1. **Drop-in Replacement**
The remote embedding function maintains the same API as `embedding.py`:
```python
score = compute_score(
    data_source="embedding_remote_test",
    solution_str="Your model output",
    ground_truth="Reference answer",
    extra_info={"server_url": "http://your-server:8080"}  # Optional
)
```

### 2. **Automatic Fallback**
If the embedding server is unavailable, the function automatically falls back to lexical similarity (difflib.SequenceMatcher) to ensure training continues.

### 3. **Efficient Batching**
The client automatically batches embedding requests for optimal throughput:
```python
scores = compute_score_batched(
    data_sources=["test1", "test2", ...],
    solution_strs=["output1", "output2", ...],
    ground_truths=["reference1", "reference2", ...],
)
```

### 4. **Connection Pooling & Retry Logic**
- HTTP connection pooling for efficient reuse
- Automatic retry with exponential backoff
- Configurable timeouts and retry attempts

### 5. **Caching**
Optional LRU cache for frequently used embeddings:
```python
extra_info = {
    "cache_enabled": True,
    "cache_size": 10000
}
```

### 6. **Length Penalties**
Same length penalty options as local embedding:
```python
extra_info = {
    "length_penalty": "ratio",  # Options: none, ratio, sqrt, log, quadratic, exponential
    "length_threshold": 1.5
}
```

## Configuration Options

### Environment Variables
- `EMBEDDING_SERVER_URL`: Default server URL
- `EMBEDDING_SERVER_API_KEY`: Optional authentication key
- `EMBEDDING_SERVER_TIMEOUT`: Request timeout in seconds (default: 30)

### Extra Info Parameters
Pass these in the `extra_info` dictionary:
```python
extra_info = {
    # Server configuration
    "server_url": "http://custom-server:8080",  # Override default server
    "api_key": "your-api-key",                  # Authentication
    "timeout": 45,                               # Request timeout
    
    # Length penalty configuration  
    "length_penalty": "ratio",                   # Penalty type
    "length_threshold": 2.0,                     # When to apply penalty
    
    # Reference filtering (same as embedding.py)
    "target_gt": "specific_answer",              # Filter to specific reference
    "filter_gt_by_prompt_token": True,           # Filter by prompt token
    "prompt": "The answer is..."                 # Used with token filtering
}
```

## Testing

### 1. Test Server Connectivity
```bash
cd tests/
python test_server_connectivity.py http://YOUR_SERVER_IP:8080
```

This will test:
- Server health check
- Single and batch embeddings
- Edge cases (empty text, long text, special characters)
- Full reward function integration

### 2. Run Unit Tests
```bash
python test_embedding_remote.py
```

### 3. Benchmark Performance
```bash
python benchmark_embedding_remote.py --server-url http://localhost:8080
```

This benchmarks:
- Single request latency
- Batch processing throughput
- Cache performance
- Comparison with local embeddings

## Network Connectivity Options

### Option 1: Kubernetes Service Discovery
```python
# In the same Kubernetes cluster
os.environ["EMBEDDING_SERVER_URL"] = "http://qwen3-embedding-service.embedding-service.svc.cluster.local:8080"
```

### Option 2: Direct IP/Hostname
```python
# Direct connection
os.environ["EMBEDDING_SERVER_URL"] = "http://10.0.0.5:8080"
```

### Option 3: External Load Balancer
```python
# Through load balancer
os.environ["EMBEDDING_SERVER_URL"] = "https://embeddings.example.com"
```

## Performance Tuning

### 1. Server-side Optimization
- Adjust `max_batch_tokens` and `max_batch_requests` in TEI
- Use float16 precision for faster inference
- Enable Flash Attention if supported

### 2. Client-side Optimization
- Enable connection pooling (default: 10 connections)
- Use batching for multiple texts
- Enable caching for repeated texts
- Adjust timeout based on network latency

### 3. Deployment Scaling
- Use Horizontal Pod Autoscaler in Kubernetes
- Deploy multiple TEI replicas behind a load balancer
- Consider regional deployments for global training

## Monitoring

### Health Checks
```bash
# Check server health
curl http://your-server:8080/health

# Get server info
curl http://your-server:8080/info
```

### Metrics
The TEI server exposes Prometheus metrics on port 9090:
- Request latency
- Batch sizes
- Queue depth
- Model loading time

### Logging
Monitor client logs for:
- Fallback warnings
- Connection errors
- Performance metrics

## Troubleshooting

### Server Won't Start
1. Check GPU availability: `nvidia-smi`
2. Check Docker logs: `docker logs qwen3-embedding-server`
3. Ensure sufficient memory (32GB recommended)

### Connection Errors
1. Verify server URL is accessible
2. Check firewall/network policies
3. Test with curl: `curl http://server:8080/health`

### Slow Performance
1. Check server GPU utilization
2. Increase batch size
3. Enable caching
4. Consider using multiple server replicas

### Fallback to Lexical
If you see "falling back to lexical similarity":
1. Check server health
2. Verify network connectivity
3. Check server logs for errors
4. Ensure model is fully loaded

## Advanced Usage

### Custom Embedding Models
To use a different model, update the TEI deployment:
```bash
# In deploy_tei_docker.sh or Kubernetes manifest
MODEL_ID="your-org/your-embedding-model"
```

### Authentication
For secure deployments:
1. Set up API key authentication in TEI
2. Pass the key via environment variable:
   ```bash
   export EMBEDDING_SERVER_API_KEY="your-secure-key"
   ```

### Multi-region Deployment
For global training:
1. Deploy TEI servers in multiple regions
2. Use a global load balancer
3. Configure client with regional endpoints

## Limitations

1. **Network Dependency**: Requires reliable network connection to embedding server
2. **Latency**: Network round-trip adds latency compared to local embeddings
3. **Single Point of Failure**: Consider deploying multiple server instances
4. **Model Size**: Qwen3-Embedding-8B requires significant GPU memory (16GB+)

## Future Enhancements

1. **gRPC Support**: Lower latency protocol
2. **Streaming API**: For very large batches
3. **Model Quantization**: 8-bit/4-bit support
4. **Edge Deployment**: Lightweight models for edge cases
5. **Multi-model Support**: Route to different models based on task
