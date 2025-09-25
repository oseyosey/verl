# Remote LLM Judge Reward Function

This module implements a remote LLM-as-a-judge reward function that connects to a vLLM server hosting models like Qwen3-32B. This design allows GPU-accelerated LLM inference without requiring GPU allocation on the reward worker.

## Overview

The remote LLM judge reward function (`llm_judge_remote.py`) mirrors the functionality of the local LLM judge reward (`llm_judge.py`) but delegates text generation to a remote vLLM server. This architecture is ideal for:

- **Resource Efficiency**: No GPU required on reward workers
- **Scalability**: Centralized LLM service can serve multiple training runs
- **Performance**: GPU-accelerated inference with Qwen3-32B
- **Flexibility**: Easy to switch between different LLM models

## Quick Start

### 1. Deploy the vLLM Server

#### Using Python Server (Recommended):
```bash
cd ddrl/utils_rl/llm_judge_deploy/
python run_vllm_server.py --model Qwen/Qwen3-32B --port 8000
```

#### Using Docker Container:
```bash
cd ddrl/utils_rl/llm_judge_deploy/
./run_vllm_docker.sh
```

### 2. Configure VERL

Set the LLM judge server URL:
```bash
# For same-node deployment
export LLM_JUDGE_SERVER_URL="http://localhost:8000"

# For multi-node deployment (use actual IP from server output)
export LLM_JUDGE_SERVER_URL="http://172.16.0.60:8000"
```

### 3. Use in VERL Training

In your VERL configuration, set the data source to use remote LLM judge:
```yaml
data_source: llm_judge_remote_your_dataset
```

Or configure a custom reward function:
```yaml
custom_reward_function:
  path: verl/utils/reward_score/llm_judge_remote.py
  name: compute_score
```

## Architecture

```
┌─────────────────────┐         HTTP/REST API      ┌──────────────────────┐
│   VERL Training     │ ─────────────────────────> │  vLLM Server         │
│   (Reward Worker)   │ <───────────────────────── │  Qwen3-32B           │
│   - No GPU needed   │      Text Generation       │  - GPU accelerated   │
└─────────────────────┘                            └──────────────────────┘
```

## Features

### 1. **Drop-in Replacement**
The remote LLM judge function maintains the same API as `llm_judge.py`:
```python
score = compute_score(
    data_source="llm_judge_remote_test",
    solution_str="Your model output",
    ground_truth="Reference answer",
    extra_info={
        "problem": "What is 2+2?",
        "server_url": "http://your-server:8000"  # Optional
    }
)
```

### 2. **Graceful Error Handling**
If the LLM server is unavailable, the function returns 0.0 scores with appropriate warnings instead of crashing training.

### 3. **Efficient Batching**
The client automatically batches LLM requests for optimal throughput:
```python
scores = compute_score_batched(
    data_sources=["test1", "test2", ...],
    solution_strs=["output1", "output2", ...],
    ground_truths=["reference1", "reference2", ...],
    extra_infos=[{"problem": "prob1"}, {"problem": "prob2"}, ...]
)
```

### 4. **Connection Pooling & Retry Logic**
- HTTP connection pooling for efficient reuse
- Automatic retry with exponential backoff
- Configurable timeouts and retry attempts

### 5. **Thinking Mode Support**
Support for Qwen3 thinking mode:
```python
extra_info = {
    "problem": "Complex reasoning problem",
    "enable_thinking": True,
    "model_name": "Qwen/Qwen3-32B"
}
```

### 6. **Multiple Prompt Templates**
Choose from different evaluation styles:
```python
# Simple, concise evaluation (default)
extra_info = {"prompt_template": "default"}

# Detailed evaluation with reasoning steps
extra_info = {"prompt_template": "detailed"}

# Custom template string
extra_info = {
    "prompt_template": """
Rate similarity between solutions.
Problem: {PROBLEM}
Reference: {REFERENCE_SOLUTION}  
Candidate: {CANDIDATE_SOLUTION}
REWARD: <score>
"""
}
```

### 7. **Reference Filtering**
Same reference filtering options as local LLM judge:
```python
extra_info = {
    "target_gt": "specific_answer",              # Filter to specific reference
    "filter_gt_by_prompt_token": True,           # Filter by prompt token
    "prompt": "The answer is..."                 # Used with token filtering
}
```

## Configuration Options

### Environment Variables
- `LLM_JUDGE_SERVER_URL`: Default server URL
- `LLM_JUDGE_SERVER_API_KEY`: Optional authentication key
- `LLM_JUDGE_SERVER_TIMEOUT`: Request timeout in seconds (default: 60)

### Extra Info Parameters
Pass these in the `extra_info` dictionary:
```python
extra_info = {
    # Server configuration
    "server_url": "http://custom-server:8000",  # Override default server
    "api_key": "your-api-key",                  # Authentication
    "timeout": 90,                              # Request timeout
    
    # LLM configuration
    "model_name": "Qwen/Qwen3-32B",            # Model on vLLM server
    "temperature": 0.7,                         # Generation temperature
    "top_p": 0.8,                              # Top-p sampling
    "max_new_tokens": 512,                      # Max tokens to generate
    "enable_thinking": False,                   # Thinking mode
    "batch_size": 128,                           # Batch size for processing
    
    # Prompt configuration
    "problem": "Math problem statement",        # Required for judge prompt
    "prompt_template": "default",               # Template name: "default", "detailed", etc.
    # OR direct template string:
    # "prompt_template": "Rate solutions... {PROBLEM} {REFERENCE_SOLUTION} {CANDIDATE_SOLUTION}"
    
    # Reference filtering (same as llm_judge.py)
    "target_gt": "specific_answer",             # Filter to specific reference
    "filter_gt_by_prompt_token": True,          # Filter by prompt token
    "prompt": "The answer is..."                # Used with token filtering
}
```

## Testing

### 1. Test Server Connectivity
```bash
cd tests/
python test_llm_judge_remote_server_integration.py --server-url http://YOUR_SERVER_IP:8000
```

This will test:
- Server health check
- Single and batch LLM judge evaluation
- Performance characteristics
- Full reward function integration

### 2. Run Unit Tests
```bash
python test_llm_judge_remote_reward.py
```

### 3. Test Without Server
```bash
python test_llm_judge_remote_reward.py --skip-server-tests
```

This tests basic functionality and fallback behavior without requiring a server.

## Data Preprocessing

### Generate Training Data
Use the provided script to generate training data with LLM judge remote rewards:

```bash
cd scripts/process_data/math500_mia/
./prepare_math500_mia_data_llm_judge_remote_qwen_same_problem.sh
```

This script:
- Processes MATH-500 dataset
- Creates member/non-member pairs for MIA evaluation
- Configures remote LLM judge reward
- Saves parquet files ready for VERL training

## Performance Considerations

### 1. Server-side Optimization
- Use multiple vLLM server instances behind a load balancer
- Configure appropriate `max_num_seqs` and `max_num_batched_tokens`
- Use efficient model precision (float16)

### 2. Client-side Optimization
- Enable connection pooling (default: 1028 connections)
- Use batching for multiple evaluations
- Adjust timeout based on model size and complexity
- Consider caching for repeated evaluations

### 3. Deployment Scaling
- Use Horizontal Pod Autoscaler in Kubernetes
- Deploy multiple vLLM replicas behind a load balancer
- Consider regional deployments for global training

## Monitoring

### Health Checks
```bash
# Check server health
curl http://your-server:8000/health

# Get model info
curl http://your-server:8000/v1/models
```

### Logging
Monitor client logs for:
- Fallback warnings
- Connection errors
- Performance metrics
- Score extraction failures

## Troubleshooting

### Server Won't Start
1. Check GPU availability: `nvidia-smi`
2. Check Docker logs: `docker logs vllm-server`
3. Ensure sufficient memory (64GB+ recommended for Qwen3-32B)

### Connection Errors
1. Verify server URL is accessible
2. Check firewall/network policies
3. Test with curl: `curl http://server:8000/health`

### Slow Performance
1. Check server GPU utilization
2. Increase batch size
3. Enable connection pooling
4. Consider using multiple server replicas

### Fallback to Lexical
If you see "falling back to lexical similarity":
1. Check server health
2. Verify network connectivity
3. Check server logs for errors
4. Ensure model is fully loaded

## Advanced Usage

### Custom LLM Models
To use a different model, update the vLLM deployment:
```bash
python run_vllm_server.py --model your-org/your-llm-model --port 8000
```

### Authentication
For secure deployments:
1. Set up API key authentication in vLLM
2. Pass the key via environment variable:
   ```bash
   export LLM_JUDGE_SERVER_API_KEY="your-secure-key"
   ```

### Multi-model Support
Deploy multiple vLLM servers with different models and route based on task:
```python
extra_info = {
    "server_url": "http://math-model-server:8000" if is_math_problem else "http://general-server:8000",
    "model_name": "specialized-math-model" if is_math_problem else "Qwen/Qwen3-32B"
}
```

## Limitations

1. **Network Dependency**: Requires reliable network connection to vLLM server
2. **Latency**: Network round-trip adds latency compared to local inference
3. **Single Point of Failure**: Consider deploying multiple server instances
4. **Model Size**: Qwen3-32B requires significant GPU memory (64GB+)
5. **Cost**: GPU server costs for dedicated inference

## Future Enhancements

1. **gRPC Support**: Lower latency protocol
2. **Streaming API**: For very large evaluations
3. **Model Quantization**: 8-bit/4-bit support for efficiency
4. **Edge Deployment**: Lightweight models for edge cases
5. **Multi-model Routing**: Automatic model selection based on task complexity
