# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GPU-enabled reward computation worker for VERL PPO training.

This module provides a Ray worker that can be allocated GPU resources
specifically for reward computation, allowing embedding models to use
GPU acceleration.
"""

import ray
from typing import List, Dict, Any
from verl import DataProto
from verl.utils.reward_score import default_compute_score


@ray.remote(num_gpus=1)  # Allocate 1 GPU for this worker
class GPURewardWorker:
    """
    Ray worker that runs reward computation on GPU.
    
    This worker is specifically designed to handle reward computation
    that requires GPU resources, such as embedding models with FlashAttention2.
    """
    
    def __init__(self, compute_score_fn=None):
        """
        Initialize the GPU reward worker.
        
        Args:
            compute_score_fn: Custom reward function to use. If None, uses default_compute_score.
        """
        self.compute_score_fn = compute_score_fn or default_compute_score
        print(f"[GPURewardWorker] Initialized with GPU access")
    
    def compute_rewards(self, data_proto: DataProto) -> List[float]:
        """
        Compute rewards for a batch of data using GPU acceleration.
        
        Args:
            data_proto: DataProto containing the batch of data to compute rewards for
            
        Returns:
            List of reward scores for each item in the batch
        """
        import torch
        print(f"[GPURewardWorker] Computing rewards with CUDA available: {torch.cuda.is_available()}")
        
        rewards = []
        for i in range(len(data_proto)):
            data_item = data_proto[i]
            
            # Extract the necessary information
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode the text (this would need the tokenizer)
            # For now, we'll use a placeholder - in practice, you'd pass the tokenizer
            prompt_str = f"prompt_{i}"  # Placeholder
            response_str = f"response_{i}"  # Placeholder
            
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get("data_source", "embedding_match_custom")
            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            
            # Compute the reward score
            score = self.compute_score_fn(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            rewards.append(float(score))
        
        return rewards
    
    def test_gpu_access(self) -> Dict[str, Any]:
        """
        Test GPU access and return information about the available GPU.
        
        Returns:
            Dictionary containing GPU information
        """
        import torch
        return {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
        }
