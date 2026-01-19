"""
Convert pre-training MIA datasets (WikiMIA-24, BookMIA) to verl RL parquet format with MIA support.

This script:
- Loads MIA datasets with text and labels (0=non-member, 1=member)
  - WikiMIA-24: uses "input" field
  - BookMIA: uses "snippet" field
- Splits input text into prefix and suffix based on prefix_ratio (word count)
- Adds configurable custom prompt to user messages
- Supports MIA data generation with pre-labeled members and non-members
- Optionally generates augmented target_gt with multiple solutions
- Saves both RL parquet file and optional MIA JSONL files

Usage:
  # WikiMIA-24 with default settings
  python pre_training_custom_mia.py --dataset_path wjfu99/WikiMIA-24 --dataset_length 256
  
  # BookMIA dataset
  python pre_training_custom_mia.py --dataset_path swj0419/BookMIA --input_field snippet
  
  # With custom prefix ratio and prompt
  python pre_training_custom_mia.py --prefix_ratio 0.50 --custom_prompt "Continue the generation as closely to verbatim as possible. "
  
  # With MIA data generation
  python pre_training_custom_mia.py --mia --subset_size 64
  
  # With target GT augmentation
  python pre_training_custom_mia.py --mia --include_target_gt --augment_target_gt --augment_sampling_method random --augment_num_samples 7
  
  # With assistant prefix
  python pre_training_custom_mia.py --enable_assistant_prefix --assistant_prefix_ratio 0.25
"""

import argparse
import os
import random
import json
from functools import partial
from typing import List, Tuple

import datasets

from verl.utils.fs import copy, makedirs  # type: ignore


def split_input_by_prefix_ratio(input_text: str, prefix_ratio: float = 0.50) -> Tuple[str, str]:
    """Split input text by word count at specified ratio.
    
    Args:
        input_text: The full input text to split
        prefix_ratio: Ratio of words to use as prefix (0.0 to 1.0)
        
    Returns:
        Tuple of (prefix, suffix) strings
    """
    words = input_text.split()
    if len(words) == 0:
        return "", ""
    
    num_prefix_words = max(1, int(len(words) * prefix_ratio))
    # Ensure we don't exceed total words
    num_prefix_words = min(num_prefix_words, len(words))
    
    prefix = " ".join(words[:num_prefix_words])
    suffix = " ".join(words[num_prefix_words:]) if num_prefix_words < len(words) else ""
    
    return prefix, suffix


def transform_example(
    example,
    idx: int,
    split: str,
    match_type: str = "lexical",
    metric: str = "bm25",
    include_target_gt: bool = False,
    verbose: bool = True,
    # Lexical metric parameters
    lexical_metric_profile: str = "default",
    lexical_custom_weights: List[float] = None,
    lexical_num_workers: int = 32,
    lexical_show_progress: bool = True,
    # Augmentation parameters
    augmented_solutions: List[str] = None,
    exclude_original_solution: bool = False,
):
    """Convert WikiMIA-24 record into verl RL parquet compatible format.

    Parameters
    ----------
    example : dict
        A single raw dataset record with 'problem' and 'solution' fields
        (already split from input text).
    idx : int
        Index within the split – used to create a stable identifier.
    split : str
        Data split name (e.g. "WikiMIA_length256"). 
    match_type : {"lexical"}
        Determines which reward type will be used at training-time.
    metric : str
        Specific similarity metric to be applied by the reward function.
    verbose : bool, default True
        Print extra information while transforming – useful for debugging.
    """

    if match_type not in {"lexical"}:
        raise ValueError(
            f"Unsupported match_type: {match_type!r}. Currently only 'lexical' is supported for pre-training."
        )

    # Construct extra_info section
    extra_info = {
        "split": split,
        "index": idx,
        "metric": metric,
    }
    
    # Add lexical-specific configuration
    if match_type == "lexical":
        extra_info["metric_profile"] = lexical_metric_profile
        
        if lexical_custom_weights is not None:
            extra_info["custom_weights"] = lexical_custom_weights
        
        extra_info["num_workers"] = lexical_num_workers
        extra_info["show_progress"] = lexical_show_progress
        
        # Map legacy metric names
        legacy_mapping = {
            "bm25": "default",
            "ratio": "default",
            "token_ratio": "default",
            "ordered_token": "default",
            "levenshtein": "default"
        }
        
        if metric in legacy_mapping and lexical_metric_profile == "default":
            extra_info["metric_profile"] = legacy_mapping[metric]
            if verbose and idx == 0:
                print(f"[transform_example] Mapped legacy metric '{metric}' to metric_profile '{legacy_mapping[metric]}'")
        
        if verbose and idx == 0:
            print(f"[transform_example] Lexical configuration: metric_profile='{extra_info['metric_profile']}', num_workers={lexical_num_workers}")
            if lexical_custom_weights:
                print(f"[transform_example] Using custom weights: {lexical_custom_weights}")
    
    # Preserve is_member flag if present
    if "is_member" in example:
        extra_info["is_member"] = example["is_member"]
    
    # Preserve MIA-related fields for proper ID tracking
    for field in ["original_idx", "label"]:
        if field in example:
            extra_info[field] = example[field]

    # Optionally include ground-truth answer as target reference
    if include_target_gt:
        base_solution = str(example["solution"]).strip()
        
        # Check if we have augmented solutions
        if augmented_solutions is not None and len(augmented_solutions) > 0:
            if exclude_original_solution:
                extra_info["target_gt"] = augmented_solutions
                if verbose and idx == 0:
                    print(f"[transform_example] Added augmented target_gt (EXCLUDING original) for first example:")
                    print(f"  - Original solution EXCLUDED")
                    print(f"  - Using {len(augmented_solutions)} augmented solution(s) only")
                    for i, aug_sol in enumerate(augmented_solutions[:2]):
                        print(f"    [{i+1}] {aug_sol[:100]}...")
            else:
                extra_info["target_gt"] = [base_solution] + augmented_solutions
                if verbose and idx == 0:
                    print(f"[transform_example] Added augmented target_gt for first example:")
                    print(f"  - Original solution: {base_solution[:100]}...")
                    print(f"  - Added {len(augmented_solutions)} additional solution(s)")
                    for i, aug_sol in enumerate(augmented_solutions[:2]):
                        print(f"    [{i+1}] {aug_sol[:100]}...")
        else:
            extra_info["target_gt"] = base_solution
            if verbose and idx == 0:
                print(f"[transform_example] Added target_gt for first example: {base_solution[:100]}...")

    # Preserve optional assistant prefix if present
    if "assistant_prefix" in example:
        if verbose and idx == 0:
            print(f"[transform_example] Adding assistant_prefix for first example: {example['assistant_prefix']}")
        extra_info["assistant_prefix"] = example["assistant_prefix"]

    # Decide on data_source
    if match_type == "lexical":
        data_source = "lexical_match_custom"
    else:
        raise ValueError(f"Unsupported match_type: {match_type!r}")

    # Build the verl record
    record = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": str(example["problem"]).strip()}],
        "ability": f"{match_type}_match",
        "reward_model": {
            "style": "model",
            "ground_truth": str(example["solution"]).strip(),
        },
        "extra_info": extra_info,
    }

    # Print structure for first sample when verbose
    if verbose and idx == 0:
        print("[transform_example] Sample transformed record structure:")
        preview_record = {
            "data_source": record["data_source"],
            "prompt": f"[{len(record['prompt'])} messages]",
            "ability": record["ability"],
            "reward_model": {
                "style": record["reward_model"]["style"],
                "ground_truth": f"{len(record['reward_model']['ground_truth'])} chars"
            },
            "extra_info": {k: f"{type(v).__name__}" for k, v in record["extra_info"].items()}
        }
        print(json.dumps(preview_record, indent=2, ensure_ascii=False))

    return record


def _write_jsonl(path: str, rows: List[dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_lexical_similarities(query_solution: str, candidate_solutions: List[str]) -> List[float]:
    """Compute Jaccard similarity between query solution and candidates.
    
    Uses regex-based tokenizer by default.
    
    Args:
        query_solution: The query solution text
        candidate_solutions: List of candidate solution texts
        
    Returns:
        List of Jaccard similarity scores [0, 1]
    """
    import os
    import re
    
    pattern = r"\\[a-zA-Z]+(?:\{[^}]*\})*|\d+\.?\d*|[a-zA-Z_]\w*|[+\-*/=<>!]=?|[(){}\[\]]|\S"
    use_transformers = os.environ.get("DDRL_USE_TRANSFORMERS_TOKENIZER", "").strip().lower() in {"1", "true", "yes", "on"}
    tokenizer = None
    if use_transformers:
        try:
            from transformers import AutoTokenizer as _AutoTokenizer
            tokenizer = _AutoTokenizer.from_pretrained("allenai/tulu-2-7b")
        except Exception:
            tokenizer = None
    
    # Tokenize query
    if tokenizer is not None:
        query_tokens = set(tokenizer.tokenize(query_solution))
    else:
        query_tokens = set(re.findall(pattern, query_solution.lower()))
    
    # Compute similarity for each candidate
    similarities = []
    for cand in candidate_solutions:
        if tokenizer is not None:
            cand_tokens = set(tokenizer.tokenize(cand))
        else:
            cand_tokens = set(re.findall(pattern, cand.lower()))
        
        # Jaccard similarity
        intersection = query_tokens & cand_tokens
        union = query_tokens | cand_tokens
        similarity = len(intersection) / len(union) if union else 0.0
        similarities.append(similarity)
    
    return similarities


# Global cache for embedding model
_EMBEDDING_MODEL_CACHE = {}

# Global cache for tokenizers
_TOKENIZER_CACHE = {}


def count_tokens(text: str, tokenizer_name: str = "allenai/tulu-2-7b") -> int:
    """Count tokens in text using a cached tokenizer.
    
    Args:
        text: The text to tokenize
        tokenizer_name: Name of the tokenizer to use
        
    Returns:
        Number of tokens in the text
    """
    if tokenizer_name not in _TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER_CACHE[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise ImportError(f"Failed to load tokenizer {tokenizer_name}: {e}")
    
    tokenizer = _TOKENIZER_CACHE[tokenizer_name]
    return len(tokenizer.encode(text))


def _get_embedding_model():
    """Get or load the embedding model (cached for efficiency)."""
    if "model" in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE["model"]
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as e:
        raise ImportError(f"sentence-transformers is required for embedding similarity: {e}")
    
    model_name = "Qwen/Qwen3-Embedding-8B"
    
    try:
        if torch.cuda.is_available():
            embedding_device = 0
            print(f"Loading {model_name} on cuda:{embedding_device} (cached for reuse)...")
            
            model = SentenceTransformer(
                model_name,
                model_kwargs={
                    "attn_implementation": "flash_attention_2",
                    "device_map": {"": f"cuda:{embedding_device}"},
                    "torch_dtype": torch.float16
                },
                tokenizer_kwargs={"padding_side": "left"},
            )
        else:
            print(f"Loading {model_name} on CPU (cached for reuse)...")
            model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Warning: Failed to load with optimizations: {e}. Using basic loading.")
        model = SentenceTransformer(model_name)
    
    _EMBEDDING_MODEL_CACHE["model"] = model
    return model


def _clear_embedding_model_cache():
    """Clear the embedding model cache and free GPU memory."""
    if "model" in _EMBEDDING_MODEL_CACHE:
        try:
            import torch
            import gc
            
            del _EMBEDDING_MODEL_CACHE["model"]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            
            print("✓ Cleared embedding model cache and freed GPU memory")
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")


def compute_embedding_similarities(query_solution: str, candidate_solutions: List[str]) -> List[float]:
    """Compute cosine similarity using Qwen3-8B embeddings.
    
    Args:
        query_solution: The query solution text
        candidate_solutions: List of candidate solution texts
        
    Returns:
        List of cosine similarity scores [0, 1]
    """
    import numpy as np
    
    model = _get_embedding_model()
    
    all_texts = [query_solution] + candidate_solutions
    embeddings = model.encode(
        all_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    query_emb = embeddings[0]
    cand_embs = embeddings[1:]
    
    similarities = cand_embs @ query_emb
    similarities = (similarities + 1.0) / 2.0
    
    return similarities.tolist()


def sample_additional_solutions(
    current_solution: str,
    current_idx: int,
    all_solutions: List[str],
    all_indices: List[Tuple],
    method: str,
    num_samples: int,
    seed: int = 42,
    paired_solution: str = None
) -> List[str]:
    """Sample additional solutions from the pool, excluding current example.
    
    Args:
        current_solution: Solution from the current example
        current_idx: Index of current example in the pool
        all_solutions: Pool of all solutions (from members + non-members)
        all_indices: Corresponding indices as tuples (type, idx)
        method: "random", "embedding", "lexical", or "perturbed"
        num_samples: Number of solutions to sample
        seed: Random seed for reproducibility
        paired_solution: For "perturbed" method, the corresponding paired solution to include first
        
    Returns:
        List of sampled solution strings
    """
    # Filter out current example
    candidate_solutions = []
    candidate_indices = []
    
    for i, (sol, idx_tuple) in enumerate(zip(all_solutions, all_indices)):
        if i != current_idx:
            candidate_solutions.append(sol)
            candidate_indices.append(i)
    
    if len(candidate_solutions) == 0:
        return []
    
    num_samples = min(num_samples, len(candidate_solutions))
    
    if method == "random":
        rng = random.Random(seed + current_idx)
        sampled_indices = rng.sample(range(len(candidate_solutions)), num_samples)
        sampled_solutions = [candidate_solutions[i] for i in sampled_indices]
        
    elif method == "embedding":
        similarities = compute_embedding_similarities(current_solution, candidate_solutions)
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        top_k_indices = sorted_indices[:num_samples]
        sampled_solutions = [candidate_solutions[i] for i in top_k_indices]
        
    elif method == "lexical":
        similarities = compute_lexical_similarities(current_solution, candidate_solutions)
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        top_k_indices = sorted_indices[:num_samples]
        sampled_solutions = [candidate_solutions[i] for i in top_k_indices]
        
    elif method == "perturbed":
        # For perturbed method, include paired solution first, then random samples
        sampled_solutions = []
        
        # Add paired solution first if provided
        if paired_solution is not None:
            sampled_solutions.append(paired_solution)
        
        # Calculate how many random samples we need
        remaining_samples = num_samples - len(sampled_solutions)
        
        if remaining_samples > 0:
            # Filter out the paired solution from candidates if it exists
            filtered_candidates = [sol for sol in candidate_solutions if sol != paired_solution]
            remaining_samples = min(remaining_samples, len(filtered_candidates))
            
            if remaining_samples > 0:
                rng = random.Random(seed + current_idx)
                sampled_indices = rng.sample(range(len(filtered_candidates)), remaining_samples)
                random_solutions = [filtered_candidates[i] for i in sampled_indices]
                sampled_solutions.extend(random_solutions)
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return sampled_solutions


def load_and_normalize_mia_weights(
    members_path: str,
    nonmembers_path: str,
    verbose: bool = False
) -> Tuple[List[float], List[float]]:
    """Load MIA weights from JSONL files and normalize them together.
    
    Args:
        members_path: Path to members JSONL file with 'idx', 'id', and 'score' fields
        nonmembers_path: Path to non-members JSONL file with 'idx', 'id', and 'score' fields
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (normalized_member_weights, normalized_nonmember_weights)
    """
    import numpy as np
    
    # Load member weights
    member_scores = []
    with open(members_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            member_scores.append((obj["idx"], obj["score"]))
    
    # Load non-member weights
    nonmember_scores = []
    with open(nonmembers_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            nonmember_scores.append((obj["idx"], obj["score"]))
    
    # Sort by idx
    member_scores.sort(key=lambda x: x[0])
    nonmember_scores.sort(key=lambda x: x[0])
    
    # Extract scores
    member_weights = [score for _, score in member_scores]
    nonmember_weights = [score for _, score in nonmember_scores]
    
    # Combine for normalization
    all_weights = np.array(member_weights + nonmember_weights)
    
    # Normalize to [0, 1]
    min_weight = np.min(all_weights)
    max_weight = np.max(all_weights)
    
    if max_weight > min_weight:
        normalized_all = (all_weights - min_weight) / (max_weight - min_weight)
    else:
        normalized_all = np.full_like(all_weights, 0.5)
    
    # Split back
    normalized_member_weights = normalized_all[:len(member_weights)].tolist()
    normalized_nonmember_weights = normalized_all[len(member_weights):].tolist()
    
    if verbose:
        print(f"[load_and_normalize_mia_weights] Loaded {len(member_weights)} member weights and {len(nonmember_weights)} non-member weights")
        print(f"[load_and_normalize_mia_weights] Original range: [{min_weight:.4f}, {max_weight:.4f}]")
        print(f"[load_and_normalize_mia_weights] Normalized range: [0.0, 1.0]")
        print(f"[load_and_normalize_mia_weights] Member weights stats: mean={np.mean(normalized_member_weights):.4f}, std={np.std(normalized_member_weights):.4f}")
        print(f"[load_and_normalize_mia_weights] Non-member weights stats: mean={np.mean(normalized_nonmember_weights):.4f}, std={np.std(normalized_nonmember_weights):.4f}")
    
    return normalized_member_weights, normalized_nonmember_weights


def main():
    parser = argparse.ArgumentParser(
        description="Convert WikiMIA-24 dataset to verl RL parquet format with optional MIA data generation."
    )
    parser.add_argument(
        "--dataset_path",
        default="wjfu99/WikiMIA-24",
        help="HuggingFace dataset path to load (default: wjfu99/WikiMIA-24)",
    )
    parser.add_argument(
        "--dataset_split",
        default=None,
        help="Dataset split to use (default: auto-detect based on dataset)",
    )
    parser.add_argument(
        "--dataset_length",
        type=int,
        choices=[32, 64, 128, 256],
        default=256,
        help="WikiMIA dataset length variant to use (only for WikiMIA-24, default: 256)",
    )
    parser.add_argument(
        "--input_field",
        default="input",
        help="Name of the field containing input text (default: 'input' for WikiMIA, use 'snippet' for BookMIA)",
    )
    parser.add_argument(
        "--prefix_ratio",
        type=float,
        default=0.50,
        help="Ratio of words to use as prefix (0.0 to 1.0, default: 0.50)",
    )
    parser.add_argument(
        "--custom_prompt",
        default="Continue the generation as closely to verbatim as possible. ",
        help="Custom prompt to prepend to user messages (default: 'Continue the generation as closely to verbatim as possible. ')",
    )
    parser.add_argument(
        "--match_type",
        choices=["lexical"],
        default="lexical",
        help="Reward type (currently only 'lexical' is supported for pre-training)",
    )
    parser.add_argument(
        "--metric",
        default="bm25",
        help="Similarity metric to store in extra_info (e.g. bm25, ratio)",
    )
    parser.add_argument(
        "--lexical_metric_profile",
        default="default",
        help="Metric profile for lexical matching.",
    )
    parser.add_argument(
        "--lexical_custom_weights",
        type=float,
        nargs="+",
        default=None,
        help="Custom weights for lexical metrics.",
    )
    parser.add_argument(
        "--lexical_num_workers",
        type=int,
        default=32,
        help="Number of parallel workers for lexical metrics computation (default: 32)",
    )
    parser.add_argument(
        "--lexical_show_progress",
        action="store_true",
        help="Show progress bar during lexical reward computation",
    )
    parser.add_argument(
        "--include_target_gt",
        action="store_true",
        help="Include ground-truth solution as 'target_gt' in extra_info",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Number of examples to sample from each label group (member and non-member). If not specified, uses all available examples.",
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="Random seed for deterministic subsampling (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/wikimia24_rl",
        help="Directory where output Parquet and JSONL files will be saved",
    )
    parser.add_argument(
        "--output_name",
        default="wikimia24_mia",
        help="Base name for MIA JSONL outputs when --mia is set",
    )
    parser.add_argument(
        "--mia",
        action="store_true",
        help="If set, generate MIA data with separate member and non-member records",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to mirror the Parquet file to",
    )
    parser.add_argument(
        "--enable_assistant_prefix",
        action="store_true",
        help="Enable assistant prefix generation from solution text",
    )
    parser.add_argument(
        "--assistant_prefix_ratio",
        type=float,
        default=0.25,
        help="Ratio of words from solution to use as assistant prefix (default: 0.25)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging",
    )
    parser.add_argument(
        "--mia_weights_members",
        default=None,
        help="Path to JSONL file containing MIA weights for member examples",
    )
    parser.add_argument(
        "--mia_weights_nonmembers",
        default=None,
        help="Path to JSONL file containing MIA weights for non-member examples",
    )
    parser.add_argument(
        "--mia_weights_tag",
        choices=["loss", "loss_ref", "min_k", "min_k++"],
        default=None,
        help="Tag identifying the type of MIA weights",
    )
    parser.add_argument(
        "--augment_target_gt",
        action="store_true",
        help="Augment target_gt with additional solutions (requires --mia and --include_target_gt)",
    )
    parser.add_argument(
        "--augment_sampling_method",
        choices=["random", "embedding", "lexical", "perturbed"],
        default="random",
        help="Method for sampling additional solutions (default: random)",
    )
    parser.add_argument(
        "--augment_num_samples",
        type=int,
        default=1,
        help="Number of additional solutions to sample (default: 1)",
    )
    parser.add_argument(
        "--diff_threshold",
        type=int,
        default=None,
        help="Filter examples by diff field (keep only diff > threshold). Only applicable to datasets with 'diff' field.",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=None,
        help="Minimum token count for input text filtering",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum token count for input text filtering",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="allenai/tulu-2-7b",
        help="Tokenizer to use for token counting (default: allenai/tulu-2-7b)",
    )
    parser.add_argument(
        "--strict_pairing",
        action="store_true",
        help="Enforce strict pairing: group by (title, id), keep only complete member/non-member pairs, and sample N pairs",
    )
    parser.add_argument(
        "--drop_empty_input",
        action="store_true",
        help="Drop examples with empty or whitespace-only input text before sampling",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.augment_target_gt:
        if not args.mia:
            raise ValueError("--augment_target_gt requires --mia to be enabled")
        if not args.include_target_gt:
            raise ValueError("--augment_target_gt requires --include_target_gt to be enabled")
        if args.augment_num_samples < 1:
            raise ValueError("--augment_num_samples must be at least 1")
        if args.augment_sampling_method == "perturbed" and not args.strict_pairing:
            raise ValueError("--augment_sampling_method='perturbed' requires --strict_pairing to be enabled")
    
    if args.mia_weights_members or args.mia_weights_nonmembers:
        if not (args.mia_weights_members and args.mia_weights_nonmembers):
            raise ValueError("Both --mia_weights_members and --mia_weights_nonmembers must be provided together")
        if not args.mia_weights_tag:
            raise ValueError("--mia_weights_tag is required when MIA weight files are provided")
        if not args.mia:
            raise ValueError("--mia flag must be enabled when using MIA weights")
        
        if not os.path.exists(args.mia_weights_members):
            raise FileNotFoundError(f"MIA weights file not found: {args.mia_weights_members}")
        if not os.path.exists(args.mia_weights_nonmembers):
            raise FileNotFoundError(f"MIA weights file not found: {args.mia_weights_nonmembers}")
    
    if args.prefix_ratio < 0.0 or args.prefix_ratio > 1.0:
        raise ValueError(f"--prefix_ratio must be between 0.0 and 1.0, got {args.prefix_ratio}")

    # Determine split name based on dataset
    if args.dataset_split is not None:
        split_name = args.dataset_split
    elif "wikiMIA-2024-hard" in args.dataset_path:
        # wikiMIA-2024-hard uses "test" split
        split_name = "test"
    elif "WikiMIA" in args.dataset_path:
        split_name = f"WikiMIA_length{args.dataset_length}"
    else:
        # Default to "train" for other datasets (e.g., BookMIA)
        split_name = "train"
    
    # Load dataset
    ds_full = datasets.load_dataset(args.dataset_path, split=split_name)
    
    if args.verbose:
        print(f"[main] Loaded {len(ds_full)} examples from {args.dataset_path} ({split_name} split)")
        if len(ds_full) > 0:
            print(f"[main] Sample fields: {list(ds_full[0].keys())}")
            if args.input_field in ds_full[0]:
                print(f"[main] Sample {args.input_field} length: {len(ds_full[0][args.input_field])} chars")
            print(f"[main] Sample label: {ds_full[0]['label']}")
    
    # Validate input field exists
    if len(ds_full) > 0 and args.input_field not in ds_full[0]:
        available_fields = list(ds_full[0].keys())
        raise ValueError(
            f"Input field '{args.input_field}' not found in dataset. "
            f"Available fields: {available_fields}. "
            f"Use --input_field to specify the correct field name."
        )
    
    # Apply filtering if specified
    filtered_indices = []
    
    if (
        args.drop_empty_input
        or args.diff_threshold is not None
        or args.min_tokens is not None
        or args.max_tokens is not None
    ):
        if args.verbose:
            print(f"\n=== Applying Filters ===")
            if args.drop_empty_input:
                print("Drop empty input: enabled")
            if args.diff_threshold is not None:
                print(f"Diff threshold: > {args.diff_threshold}")
            if args.min_tokens is not None or args.max_tokens is not None:
                print(f"Token range: [{args.min_tokens or 'no min'}, {args.max_tokens or 'no max'}]")
                print(f"Using tokenizer: {args.tokenizer_name}")
        
        for i in range(len(ds_full)):
            ex = ds_full[i]
            input_text = str(ex[args.input_field]).strip()
            
            # Drop empty inputs early
            if args.drop_empty_input and not input_text:
                continue
            
            # Apply diff filtering
            if args.diff_threshold is not None:
                if 'diff' not in ex:
                    raise ValueError("Dataset does not have 'diff' field, but --diff_threshold was specified")
                if ex['diff'] <= args.diff_threshold:
                    continue
            
            # Apply token filtering
            if args.min_tokens is not None or args.max_tokens is not None:
                token_count = count_tokens(input_text, args.tokenizer_name)
                
                if args.min_tokens is not None and token_count < args.min_tokens:
                    continue
                if args.max_tokens is not None and token_count > args.max_tokens:
                    continue
            
            filtered_indices.append(i)
        
        if args.verbose:
            print(f"Filtered dataset: {len(filtered_indices)} / {len(ds_full)} examples passed filters")
    else:
        filtered_indices = list(range(len(ds_full)))
    
    # Handle strict pairing mode
    if args.strict_pairing:
        if args.verbose:
            print(f"\n=== Strict Pairing Mode ===")
        
        # Check if dataset has required fields for pairing
        if len(ds_full) > 0:
            sample_ex = ds_full[filtered_indices[0]] if filtered_indices else ds_full[0]
            if 'title' not in sample_ex or 'id' not in sample_ex:
                raise ValueError("Strict pairing requires 'title' and 'id' fields in the dataset")
        
        # Group by (title, id)
        pairs = {}
        for idx in filtered_indices:
            ex = ds_full[idx]
            key = (str(ex['title']), int(ex['id']))
            
            if key not in pairs:
                pairs[key] = {'member': None, 'non_member': None, 'member_idx': None, 'non_member_idx': None}
            
            if ex['label'] == 1:
                pairs[key]['member'] = ex
                pairs[key]['member_idx'] = idx
            else:
                pairs[key]['non_member'] = ex
                pairs[key]['non_member_idx'] = idx
        
        # Keep only complete pairs
        complete_pairs = [p for p in pairs.values() if p['member'] is not None and p['non_member'] is not None]
        
        if args.verbose:
            print(f"Found {len(complete_pairs)} complete pairs (from {len(pairs)} unique (title, id) combinations)")
        
        if len(complete_pairs) == 0:
            raise ValueError("No complete member/non-member pairs found after filtering")
        
        # Sample N pairs
        if args.subset_size is not None:
            if args.subset_size > len(complete_pairs):
                print(f"WARNING: Requested subset_size {args.subset_size} exceeds available pairs {len(complete_pairs)}. Using all {len(complete_pairs)} pairs.")
                args.subset_size = len(complete_pairs)
            
            rng = random.Random(args.subset_seed)
            sampled_pairs = rng.sample(complete_pairs, min(args.subset_size, len(complete_pairs)))
        else:
            sampled_pairs = complete_pairs
        
        # Extract indices
        sampled_members_indices = [p['member_idx'] for p in sampled_pairs]
        sampled_nonmembers_indices = [p['non_member_idx'] for p in sampled_pairs]
        
        if args.verbose:
            print(f"Sampled {len(sampled_pairs)} pairs: {len(sampled_members_indices)} members + {len(sampled_nonmembers_indices)} non-members")
            if len(sampled_pairs) > 0:
                sample_pair = sampled_pairs[0]
                print(f"Sample pair: member_idx={sample_pair['member_idx']}, non_member_idx={sample_pair['non_member_idx']}")
                print(f"  Title: {sample_pair['member']['title']}")
                print(f"  ID: {sample_pair['member']['id']}")
    
    else:
        # Original behavior: split by label and sample independently
        members_indices = [i for i in filtered_indices if ds_full[i]["label"] == 1]
        nonmembers_indices = [i for i in filtered_indices if ds_full[i]["label"] == 0]
        
        if args.verbose:
            print(f"[main] Found {len(members_indices)} members (label=1) and {len(nonmembers_indices)} non-members (label=0)")
        
        # Sample from each group if subset_size is specified
        if args.subset_size is not None:
            if args.subset_size > len(members_indices):
                print(f"WARNING: Requested subset_size {args.subset_size} exceeds available members {len(members_indices)}. Using all {len(members_indices)} members.")
                args.subset_size = len(members_indices)
            
            if args.subset_size > len(nonmembers_indices):
                print(f"WARNING: Requested subset_size {args.subset_size} exceeds available non-members {len(nonmembers_indices)}. Using all {len(nonmembers_indices)} non-members.")
                args.subset_size = len(nonmembers_indices)
            
            rng = random.Random(args.subset_seed)
            sampled_members_indices = rng.sample(members_indices, min(args.subset_size, len(members_indices)))
            sampled_nonmembers_indices = rng.sample(nonmembers_indices, min(args.subset_size, len(nonmembers_indices)))
            
            sampled_members_indices.sort()
            sampled_nonmembers_indices.sort()
            
            if args.verbose:
                print(f"[main] Sampled {len(sampled_members_indices)} members and {len(sampled_nonmembers_indices)} non-members using seed {args.subset_seed}")
        else:
            sampled_members_indices = members_indices
            sampled_nonmembers_indices = nonmembers_indices
            if args.verbose:
                print(f"[main] Using all {len(sampled_members_indices)} members and {len(sampled_nonmembers_indices)} non-members")
    
    # Create processed examples with split input
    def create_example_with_split(example, idx, is_member: bool):
        """Split input and create problem/solution structure."""
        input_text = str(example[args.input_field]).strip()
        prefix, suffix = split_input_by_prefix_ratio(input_text, args.prefix_ratio)
        
        # Add custom prompt to prefix
        problem = args.custom_prompt + prefix
        solution = suffix
        
        new_example = {
            "problem": problem,
            "solution": solution,
            "input": input_text,
            "label": example["label"],
            "is_member": is_member,
            "original_idx": idx,
        }
        
        # Add assistant prefix if enabled
        if args.enable_assistant_prefix and solution:
            words = solution.split()
            num_prefix_words = max(1, int(len(words) * args.assistant_prefix_ratio))
            assistant_prefix = " ".join(words[:num_prefix_words])
            new_example["assistant_prefix"] = assistant_prefix
        
        return new_example
    
    # Process members
    processed_members = []
    for idx in sampled_members_indices:
        ex = ds_full[idx]
        processed_ex = create_example_with_split(ex, idx, is_member=True)
        processed_members.append(processed_ex)
    
    # Process non-members
    processed_nonmembers = []
    for idx in sampled_nonmembers_indices:
        ex = ds_full[idx]
        processed_ex = create_example_with_split(ex, idx, is_member=False)
        processed_nonmembers.append(processed_ex)
    
    if args.verbose:
        print(f"[main] Processed {len(processed_members)} members and {len(processed_nonmembers)} non-members")
        if len(processed_members) > 0:
            sample_member = processed_members[0]
            print(f"[main] Sample member problem (first 100 chars): {sample_member['problem'][:100]}...")
            print(f"[main] Sample member solution (first 100 chars): {sample_member['solution'][:100]}...")
    
    # Prepare augmentation if enabled
    augmentation_map = {}
    if args.augment_target_gt:
        print(f"\n=== Target GT Augmentation ===")
        print(f"Sampling method: {args.augment_sampling_method}")
        print(f"Number of samples per example: {args.augment_num_samples}")
        
        # Collect all solutions
        all_solutions = []
        all_indices = []
        
        for i, ex in enumerate(processed_members):
            all_solutions.append(ex["solution"])
            all_indices.append(("member", i))
        
        for i, ex in enumerate(processed_nonmembers):
            all_solutions.append(ex["solution"])
            all_indices.append(("non_member", i))
        
        print(f"Total solution pool size: {len(all_solutions)} (members + non-members)")
        
        # Build pairing map for "perturbed" method
        pairing_map = {}
        if args.augment_sampling_method == "perturbed":
            # For perturbed method, we need to map each member to its corresponding non-member
            # This assumes members and non-members are in the same order (from strict pairing)
            if len(processed_members) == len(processed_nonmembers):
                for i in range(len(processed_members)):
                    # Member at index i pairs with non-member at index i
                    pairing_map[("member", i)] = processed_nonmembers[i]["solution"]
                    pairing_map[("non_member", i)] = processed_members[i]["solution"]
                
                print(f"Built pairing map with {len(pairing_map)} entries (perturbed augmentation)")
            else:
                print(f"WARNING: Cannot use perturbed augmentation with unequal member/non-member counts. Falling back to random.")
                args.augment_sampling_method = "random"
        
        # Sample for members
        for i in range(len(processed_members)):
            current_solution = all_solutions[i]
            paired_solution = pairing_map.get(("member", i), None)
            
            sampled = sample_additional_solutions(
                current_solution=current_solution,
                current_idx=i,
                all_solutions=all_solutions,
                all_indices=all_indices,
                method=args.augment_sampling_method,
                num_samples=args.augment_num_samples,
                seed=args.subset_seed,
                paired_solution=paired_solution
            )
            if sampled:
                augmentation_map[("member", i)] = sampled
        
        # Sample for non-members
        for i in range(len(processed_nonmembers)):
            current_solution = all_solutions[len(processed_members) + i]
            pool_idx = len(processed_members) + i
            paired_solution = pairing_map.get(("non_member", i), None)
            
            sampled = sample_additional_solutions(
                current_solution=current_solution,
                current_idx=pool_idx,
                all_solutions=all_solutions,
                all_indices=all_indices,
                method=args.augment_sampling_method,
                num_samples=args.augment_num_samples,
                seed=args.subset_seed,
                paired_solution=paired_solution
            )
            if sampled:
                augmentation_map[("non_member", i)] = sampled
        
        print(f"✅ Augmented {len(augmentation_map)} examples with additional solutions")
        
        # Clear embedding cache if used
        if args.augment_sampling_method == "embedding":
            _clear_embedding_model_cache()
    
    # Transform examples
    def transform_with_augmentation(example, idx, example_type):
        """Transform with optional augmentation."""
        aug_sols = augmentation_map.get((example_type, idx), None) if augmentation_map else None
        return transform_example(
            example=example,
            idx=idx,
            split=split_name,
            match_type=args.match_type,
            metric=args.metric,
            include_target_gt=args.include_target_gt,
            verbose=args.verbose,
            lexical_metric_profile=args.lexical_metric_profile,
            lexical_custom_weights=args.lexical_custom_weights,
            lexical_num_workers=args.lexical_num_workers,
            lexical_show_progress=args.lexical_show_progress,
            augmented_solutions=aug_sols,
            exclude_original_solution=False,
        )
    
    # Create datasets
    ds_members_list = [transform_with_augmentation(ex, i, "member") for i, ex in enumerate(processed_members)]
    ds_nonmembers_list = [transform_with_augmentation(ex, i, "non_member") for i, ex in enumerate(processed_nonmembers)]
    
    ds_members = datasets.Dataset.from_list(ds_members_list)
    ds_nonmembers = datasets.Dataset.from_list(ds_nonmembers_list)
    
    if args.verbose:
        print(f"[main] Member dataset: {len(ds_members)} records")
        print(f"[main] Non-member dataset: {len(ds_nonmembers)} records")
    
    # Load and apply MIA weights if provided
    if args.mia_weights_members and args.mia_weights_nonmembers:
        print(f"\n=== Loading MIA weights ({args.mia_weights_tag}) ===")
        normalized_member_weights, normalized_nonmember_weights = load_and_normalize_mia_weights(
            args.mia_weights_members,
            args.mia_weights_nonmembers,
            verbose=args.verbose
        )
        
        if len(normalized_member_weights) != len(ds_members):
            raise ValueError(
                f"Number of member weights ({len(normalized_member_weights)}) does not match "
                f"number of member examples ({len(ds_members)})"
            )
        if len(normalized_nonmember_weights) != len(ds_nonmembers):
            raise ValueError(
                f"Number of non-member weights ({len(normalized_nonmember_weights)}) does not match "
                f"number of non-member examples ({len(ds_nonmembers)})"
            )
        
        # Add weights to extra_info
        def add_mia_weight_member(example, idx):
            new_example = dict(example)
            if "extra_info" in new_example and isinstance(new_example["extra_info"], dict):
                new_example["extra_info"] = dict(new_example["extra_info"])
                new_example["extra_info"]["mia_weight"] = normalized_member_weights[idx]
                new_example["extra_info"]["mia_weight_tag"] = args.mia_weights_tag
            return new_example
        
        def add_mia_weight_nonmember(example, idx):
            new_example = dict(example)
            if "extra_info" in new_example and isinstance(new_example["extra_info"], dict):
                new_example["extra_info"] = dict(new_example["extra_info"])
                new_example["extra_info"]["mia_weight"] = normalized_nonmember_weights[idx]
                new_example["extra_info"]["mia_weight_tag"] = args.mia_weights_tag
            return new_example
        
        ds_members = ds_members.map(add_mia_weight_member, with_indices=True)
        ds_nonmembers = ds_nonmembers.map(add_mia_weight_nonmember, with_indices=True)
        
        print(f"✅ Added MIA weights ({args.mia_weights_tag}) to {len(ds_members)} members and {len(ds_nonmembers)} non-members")
    
    # Combine and shuffle
    if args.mia:
        ds_combined = datasets.concatenate_datasets([ds_members, ds_nonmembers])
        ds_transformed = ds_combined.shuffle(seed=args.subset_seed)
        
        if args.verbose:
            print(f"[main] Combined dataset: {len(ds_members)} members + {len(ds_nonmembers)} non-members = {len(ds_transformed)} total")
            member_count = sum(1 for record in ds_transformed if record.get('extra_info', {}).get('is_member', False))
            non_member_count = len(ds_transformed) - member_count
            print(f"[main] MIA data: {member_count} members, {non_member_count} non-members")
    else:
        # Only use members if MIA is not enabled
        ds_transformed = ds_members
        if args.verbose:
            print(f"[main] Using only member data: {len(ds_transformed)} records")
    
    # Save to parquet
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.parquet")
    ds_transformed.to_parquet(out_path)
    print(f"Parquet file saved at {out_path}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir:
        hdfs_path = os.path.join(args.hdfs_dir, "train.parquet")
        makedirs(args.hdfs_dir)
        copy(out_path, hdfs_path)
        print(f"Also copied to HDFS: {hdfs_path}")
    
    # Generate MIA JSONL files if requested
    if args.mia:
        members_path = os.path.join(output_dir, f"{args.output_name}_members.jsonl")
        nonmembers_path = os.path.join(output_dir, f"{args.output_name}_nonmembers.jsonl")
        indices_path = os.path.join(output_dir, f"{args.output_name}_indices.json")
        
        print(f"\n=== Generating MIA JSONL files ===")
        
        # Build member rows
        members_rows = []
        for i, ex in enumerate(processed_members):
            row = {
                "id": i,
                "problem": str(ex["problem"]).strip(),
                "solution": str(ex["solution"]).strip(),
                "input": str(ex["input"]).strip(),
                "label": ex["label"],
                "original_idx": ex["original_idx"],
                "is_member": True,
            }
            members_rows.append(row)
        
        # Build non-member rows
        nonmembers_rows = []
        for i, ex in enumerate(processed_nonmembers):
            row = {
                "id": i,
                "problem": str(ex["problem"]).strip(),
                "solution": str(ex["solution"]).strip(),
                "input": str(ex["input"]).strip(),
                "label": ex["label"],
                "original_idx": ex["original_idx"],
                "is_member": False,
            }
            nonmembers_rows.append(row)
        
        _write_jsonl(members_path, members_rows)
        _write_jsonl(nonmembers_path, nonmembers_rows)
        
        # Save indices
        # Calculate total members and nonmembers from the filtered dataset
        total_members = sum(1 for i in filtered_indices if ds_full[i]["label"] == 1)
        total_nonmembers = sum(1 for i in filtered_indices if ds_full[i]["label"] == 0)
        
        index_info = {
            "member_indices": sampled_members_indices,
            "nonmember_indices": sampled_nonmembers_indices,
            "member_seed": args.subset_seed,
            "member_size": len(sampled_members_indices),
            "nonmember_size": len(sampled_nonmembers_indices),
            "dataset_info": {
                "dataset_path": args.dataset_path,
                "dataset_length": args.dataset_length,
                "split": split_name,
                "total_dataset_size": len(ds_full),
                "total_members": total_members,
                "total_nonmembers": total_nonmembers,
            },
            "preprocessing_config": {
                "prefix_ratio": args.prefix_ratio,
                "custom_prompt": args.custom_prompt,
                "enable_assistant_prefix": args.enable_assistant_prefix,
                "assistant_prefix_ratio": args.assistant_prefix_ratio if args.enable_assistant_prefix else None,
            }
        }
        
        with open(indices_path, "w") as f:
            json.dump(index_info, f, indent=2)
        
        print(f"Wrote MIA JSONL files:")
        print(f"  Members ({len(members_rows)} examples): {members_path}")
        print(f"  Non-members ({len(nonmembers_rows)} examples): {nonmembers_path}")
        print(f"  Indices: {indices_path}")
        
        if args.verbose:
            print(f"\n=== MIA Data Verification ===")
            print(f"Members: {len(members_rows)} examples (label=1)")
            print(f"Non-members: {len(nonmembers_rows)} examples (label=0)")
            if len(members_rows) > 0:
                sample_member = members_rows[0]
                print(f"Sample member problem: {sample_member['problem'][:80]}...")
            if len(nonmembers_rows) > 0:
                sample_nonmember = nonmembers_rows[0]
                print(f"Sample non-member problem: {sample_nonmember['problem'][:80]}...")


if __name__ == "__main__":
    main()

