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
Generate responses given a dataset of prompts
"""

import copy
import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = OmegaConf.select(config, "model.trust_remote_code")
    if trust_remote_code is None:
        trust_remote_code = OmegaConf.select(config, "data.trust_remote_code")
    if trust_remote_code is None:
        trust_remote_code = False
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # * Base Model Chat Template * #
    # If using a base model without a built-in chat template, allow providing one via config
    # and otherwise fall back to a simple ChatML-style template so RL datasets that expect
    # conversational formatting can tokenize correctly.
    chat_template_cfg = OmegaConf.select(config, "data.chat_template")
    try:
        needs_template = (getattr(tokenizer, "chat_template", None) in (None, ""))
    except Exception:
        needs_template = True

    if needs_template:
        template_text = None
        if chat_template_cfg:
            # If a path is provided, read the template from file; otherwise treat as raw template text
            try:
                if isinstance(chat_template_cfg, str) and os.path.isfile(chat_template_cfg):
                    with open(chat_template_cfg, "r") as f:
                        template_text = f.read()
                else:
                    template_text = str(chat_template_cfg)
            except Exception:
                template_text = str(chat_template_cfg)
        else:
            # Minimal generic template compatible with HF chat templating
            # Jinja template for the chat template
            template_text = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}System: {{ message['content'] }}\n"
                "{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n"
                "{% elif message['role'] == 'tool' %}Tool: {{ message['content'] }}\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}Assistant:{% endif %}"
            )

        try:
            tokenizer.chat_template = template_text
        except Exception:
            # If setting fails, leave as-is; downstream will still raise a clear error
            pass

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]
    
    # Handle assistant prefix if configured
    assistant_prefix_key = config.data.get("assistant_prefix_key")
    if assistant_prefix_key:
        print(f"Processing assistant prefix with key: {assistant_prefix_key}")
        processed_chat_lst = []
        prefix_count = 0
        for i, messages in enumerate(chat_lst):
            # Check for assistant prefix in extra_info
            assistant_prefix = None
            if "extra_info" in dataset.columns:
                extra_info = dataset.iloc[i]["extra_info"]
                if isinstance(extra_info, dict) and assistant_prefix_key in extra_info:
                    assistant_prefix = extra_info[assistant_prefix_key]
            
            # If assistant prefix found, append it to messages
            if assistant_prefix:
                messages_with_prefix = copy.deepcopy(messages)
                messages_with_prefix.append({"role": "assistant", "content": assistant_prefix})
                processed_chat_lst.append(messages_with_prefix)
                prefix_count += 1
                if prefix_count <= 3:  # Show first 3 examples
                    print(f"  Example {prefix_count}: Added assistant prefix: '{assistant_prefix[:50]}...'")
            else:
                processed_chat_lst.append(messages)
        chat_lst = processed_chat_lst
        print(f"Total examples with assistant prefix: {prefix_count}/{len(chat_lst)}")
    else:
        print("No assistant prefix key configured")

    # Optionally filter out prompts that exceed a maximum token length, similar to RLHFDataset
    if OmegaConf.select(config, "data.filter_overlong_prompts"):
        max_prompt_length = OmegaConf.select(config, "data.max_prompt_length") or 1024

        def conversation_token_length(conversation_messages) -> int:
            # Build raw prompt string using the chat template, then tokenize to count tokens
            # Use same template kwargs logic as in the main generation loop
            template_kwargs = {"add_generation_prompt": True}
            if assistant_prefix_key:
                has_assistant_prefix = (
                    len(conversation_messages) > 0 and 
                    conversation_messages[-1].get("role") == "assistant"
                )
                if has_assistant_prefix:
                    template_kwargs = {"add_generation_prompt": False, "continue_final_message": True}
            
            raw_prompt = tokenizer.apply_chat_template(
                conversation_messages,
                tokenize=False,
                **template_kwargs,
            )
            return len(tokenizer(raw_prompt, add_special_tokens=False)["input_ids"])

        keep_mask = []
        for messages in chat_lst:
            try:
                keep_mask.append(conversation_token_length(messages) <= max_prompt_length)
            except Exception:
                # If any failure occurs in templating/tokenization, conservatively keep the sample
                keep_mask.append(True)

        before_n = len(chat_lst)
        chat_lst = [c for c, k in zip(chat_lst, keep_mask) if k]
        dataset = dataset.loc[keep_mask].reset_index(drop=True)
        print(f"Filtering prompts longer than {max_prompt_length} tokens: {before_n} -> {len(chat_lst)}")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    max_colocate_count = OmegaConf.select(config, "trainer.max_colocate_count")
    if max_colocate_count is None:
        max_colocate_count = 1
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        max_colocate_count=max_colocate_count,
    )
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        # Determine template kwargs based on whether assistant prefix is used
        template_kwargs = {"add_generation_prompt": True}
        if assistant_prefix_key:
            # Check if any message in the batch has assistant prefix (ends with assistant role)
            has_assistant_prefix = any(
                len(messages) > 0 and messages[-1].get("role") == "assistant" 
                for messages in batch_chat_lst
            )
            if has_assistant_prefix:
                template_kwargs = {"add_generation_prompt": False, "continue_final_message": True}
        
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            **template_kwargs,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)

            output_lst[n_sample].extend(output_texts)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset["responses"] = output_lst

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
