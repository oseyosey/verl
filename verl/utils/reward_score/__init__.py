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
# from . import gsm8k, math, prime_math, prime_code

import pdb

from verl.utils.import_utils import deprecated

def default_compute_score(
    data_source=None, 
    solution_str=None, 
    ground_truth=None, 
    extra_info=None, 
    sandbox_fusion_url=None, 
    concurrent_semaphore=None,
    # Batched arguments (for BatchRewardManager compatibility)
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    # Additional reward kwargs (passed through from config)
    **reward_kwargs,
):
    """Compute the score for a given solution based on the data source.
    
    Supports both single-sample and batched interfaces:
    - Single: data_source, solution_str, ground_truth, extra_info
    - Batched: data_sources, solution_strs, ground_truths, extra_infos

    Args:
        data_source (str, optional): The source dataset identifier which determines the scoring method.
        solution_str (str, optional): The solution string to be evaluated.
        ground_truth (str, optional): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
        data_sources (List[str], optional): List of source dataset identifiers for batch processing.
        solution_strs (List[str], optional): List of solution strings to be evaluated.
        ground_truths (List[str], optional): List of ground truth answers for comparison.
        extra_infos (List[dict], optional): List of additional information for batch processing.

    Returns:
        float or List[float]: The computed score(s) as floating point number(s). If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    # Handle batched interface
    if data_sources is not None:
        return _default_compute_score_batched(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            **reward_kwargs,
        )
    
    # Handle single interface (original logic)
    # breakpoint() # to check for data source # TODO: figure out how to debug in ray cluster.
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(sandbox_fusion_url, concurrent_semaphore, solution_str, ground_truth, continuous=True)
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa", "searchR1_hotpotqa", "searchR1_2wikimultihopqa", "searchR1_musique", "searchR1_bamboogle"]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    elif data_source.startswith("embedding_match"):
        # Embedding-based semantic similarity reward (FastText etc.)
        from . import embedding

        # Users may still override via extra_info, but we pass the metric flag
        metric_from_extra = None
        if isinstance(extra_info, dict):
            metric_from_extra = extra_info.get("metric") or extra_info.get("lexical_metric")

        res = embedding.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    elif data_source.startswith("embedding_remote"):
        # Remote embedding-based semantic similarity reward (TEI server)
        from . import embedding_remote

        res = embedding_remote.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    elif data_source.startswith("lexical_match"):
        
        # Generic lexical similarity reward with flexible metric profiles
        from . import lexical
        
        # The metric_profile is now configured in extra_info during data preprocessing
        # For backward compatibility, map old 'metric' to 'metric_profile' if needed
        if isinstance(extra_info, dict) and "metric_profile" not in extra_info:
            legacy_metric = extra_info.get("metric") or extra_info.get("lexical_metric")
            if legacy_metric:
                # Map legacy metric names to new profiles
                legacy_mapping = {
                    "bm25": "default",
                    "ratio": "default",
                    "token_ratio": "default",
                    "ordered_token": "default",
                    "levenshtein": "default"
                }
                extra_info["metric_profile"] = legacy_mapping.get(legacy_metric, "default")

        res = lexical.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **reward_kwargs,
        )
    elif data_source.startswith("llm_judge_remote"):
        # Remote LLM-as-a-judge reward using vLLM server to evaluate similarity
        from . import llm_judge_remote

        res = llm_judge_remote.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    elif data_source.startswith("llm_judge"):
        # LLM-as-a-judge reward using API calls to evaluate similarity
        from . import llm_judge

        res = llm_judge.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    elif data_source.startswith("bleurt_match"):
        # BLEURT-based semantic similarity reward
        from . import bleurt

        res = bleurt.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


def _default_compute_score_batched(
    data_sources, 
    solution_strs, 
    ground_truths, 
    extra_infos=None, 
    sandbox_fusion_url=None, 
    concurrent_semaphore=None,
    **reward_kwargs,
):
    """
    Batched version of default_compute_score for BatchRewardManager compatibility.
    
    This function routes to the appropriate batched implementations where available,
    or falls back to sequential processing for unsupported data sources.
    """
    # Handle None or empty inputs (use len() to avoid array truth value ambiguity)
    if data_sources is None or len(data_sources) == 0:
        raise ValueError("data_sources must be provided for batched processing")
    if solution_strs is None or len(solution_strs) == 0:
        raise ValueError("solution_strs must be provided for batched processing")
    if ground_truths is None or len(ground_truths) == 0:
        raise ValueError("ground_truths must be provided for batched processing")
    
    if len(data_sources) != len(solution_strs) or len(data_sources) != len(ground_truths):
        raise ValueError("data_sources, solution_strs, and ground_truths must have the same length")
    
    # Handle extra_infos default
    if extra_infos is None:
        extra_infos = [None] * len(data_sources)
    elif len(extra_infos) != len(data_sources):
        raise ValueError("extra_infos must have the same length as data_sources")
    
    # Check if all data sources are the same and support native batching
    unique_sources = set(data_sources)
    
    if len(unique_sources) == 1:
        data_source = next(iter(unique_sources))
        
        # Use native batched implementations where available
        if data_source.startswith("llm_judge_remote"):
            from . import llm_judge_remote
            return llm_judge_remote.compute_score(
                data_sources=data_sources,
                solution_strs=solution_strs,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
            )
        elif data_source.startswith("embedding_remote"):
            from . import embedding_remote
            return embedding_remote.compute_score(
                data_sources=data_sources,
                solution_strs=solution_strs,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
            )
        elif data_source.startswith("lexical_match"):
            from . import lexical
            return lexical.compute_score(
                data_sources=data_sources,
                solution_strs=solution_strs,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
                **reward_kwargs,
            )
        # Add more batched implementations here as needed
    
    # Fallback: sequential processing for mixed data sources or unsupported batching
    results = []
    for i in range(len(data_sources)):
        result = default_compute_score(
            data_source=data_sources[i],
            solution_str=solution_strs[i],
            ground_truth=ground_truths[i],
            extra_info=extra_infos[i],
            sandbox_fusion_url=sandbox_fusion_url,
            concurrent_semaphore=concurrent_semaphore,
            **reward_kwargs,
        )
        results.append(result)
    
    return results


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore)


__all__ = ["default_compute_score"]
