"""
Convert HuggingFaceH4/MATH-500 dataset to verl RL parquet format with MIA support.

This script:
- Loads MATH-500 dataset and converts to verl RL format
- Supports both lexical and embedding match types
- Optionally generates MIA (Membership Inference Attack) data with two methods:
  - Members: Original problem-solution pairs from the dataset (used for fine-tuning)
  - Non-members: Either randomly paired problems/solutions OR unused examples from dataset
- Saves both RL parquet file and optional MIA JSONL files

Usage:
  # Basic conversion without MIA
  python math500_match_custom_mia.py
  
  # With MIA data generation (safer unused examples method)
  python math500_match_custom_mia.py --mia --mia_nonmember_method unused_examples
  
  # With MIA data generation (original random pairing method)
  python math500_match_custom_mia.py --mia --mia_nonmember_method random_pairing --random_pairing_mode full_random
  
  # With MIA data generation (same problem with random solutions)
  python math500_match_custom_mia.py --mia --mia_nonmember_method random_pairing --random_pairing_mode same_problem
  
  # Custom subset size and local embedding matching
  python math500_match_custom_mia.py --subset_size 300 --match_type embedding --mia
  
  # Remote embedding matching (requires TEI server)
  python math500_match_custom_mia.py --subset_size 300 --match_type embedding_remote --mia
  
  # LLM judge with custom prompt template and thinking mode
  python math500_match_custom_mia.py --match_type llm_judge --llm_prompt_template detailed_rubric --subset_size 100
  
  # LLM judge without thinking mode (faster, lower cost)
  python math500_match_custom_mia.py --match_type llm_judge --no_llm_thinking --llm_max_tokens 256 --subset_size 100
"""

import argparse
import os
import random
import json
from functools import partial
from typing import List, Tuple

import datasets

from verl.utils.fs import copy, makedirs  # type: ignore

try:
    from .llm_judge_prompts import get_prompt_template, list_available_templates
except ImportError:
    # Handle case when running as script directly
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from llm_judge_prompts import get_prompt_template, list_available_templates


def transform_example(
    example,
    idx: int,
    split: str,
    match_type: str = "lexical",
    metric: str = "bm25",
    include_target_gt: bool = False,
    verbose: bool = True,
    llm_model: str = "gemini/gemini-2.5-flash",
    llm_temperature: float = 0.6,
    llm_max_tokens: int = 4096,
    llm_timeout: int = 30,
    llm_prompt_template: str = "default",
    llm_thinking_enabled: bool = True,
    llm_thinking_budget: int = 2048,
    # BLEURT-specific parameters
    bleurt_checkpoint: str = "lucadiliello/BLEURT-20",
    bleurt_length_penalty: str = "none",
    bleurt_length_threshold: float = 1.5,
    bleurt_device: str = None,
    # Embedding-specific parameters
    embedding_length_penalty: str = "none",
    embedding_length_threshold: float = 1.5,
):
    """Convert MATH-500 record into verl RL parquet compatible format.

    Parameters
    ----------
    example : dict
        A single raw dataset record coming from *HuggingFaceH4/MATH-500*.
    idx : int
        Index within the split – used to create a stable identifier.
    split : str
        Data split name (e.g. "test"). 
    match_type : {"lexical", "embedding", "embedding_remote", "llm_judge", "bleurt"}
        Determines which reward type will be used at training-time and thus
        controls the ``data_source`` string expected by
        ``verl.utils.reward_score.default_compute_score``.
    metric : str
        Specific similarity metric to be applied by the reward function.
    verbose : bool, default True
        Print extra information while transforming – useful for debugging.
    """

    if match_type not in {"lexical", "embedding", "embedding_remote", "llm_judge", "bleurt"}:
        raise ValueError(
            f"Unsupported match_type: {match_type!r}. Choose 'lexical', 'embedding', 'embedding_remote', 'llm_judge', or 'bleurt'."
        )

    # Construct *extra_info* section – accessible by the reward loader so that
    # it can forward *metric* to the scoring function.
    extra_info = {
        "split": split,
        "index": idx,
        "metric": metric,
    }
    
    # Preserve is_member flag if present in the example
    if "is_member" in example:
        extra_info["is_member"] = example["is_member"]
    
    # Preserve other MIA-related fields for proper ID tracking
    for field in ["original_idx", "original_problem_idx", "original_solution_idx"]:
        if field in example:
            extra_info[field] = example[field]

    # Optionally include the ground-truth answer as a dedicated target reference
    # for reward functions that support the 'target_gt' hint.
    if include_target_gt:
        if verbose:
            print(
                f"[transform_example] Added target_gt for idx={idx}: {str(example['solution']).strip()[:100]}"
            )
        extra_info["target_gt"] = str(example["solution"]).strip()

    # Preserve optional assistant prefix if upstream pipeline inserted one.
    if "assistant_prefix" in example:
        if verbose:
            print(
                f"[transform_example] Adding assistant_prefix for idx={idx}: {example['assistant_prefix']}"
            )
        extra_info["assistant_prefix"] = example["assistant_prefix"]

    # Add LLM judge specific configuration
    if match_type == "llm_judge":
        # Add problem context for LLM judge prompt formatting
        extra_info["problem"] = str(example["problem"]).strip()
        
        # Add LLM configuration
        extra_info["model"] = llm_model
        extra_info["temperature"] = llm_temperature
        extra_info["max_tokens"] = llm_max_tokens
        extra_info["timeout"] = llm_timeout
        extra_info["thinking_enabled"] = llm_thinking_enabled
        extra_info["thinking_budget"] = llm_thinking_budget
        
        # Add prompt template for LLM judge from configuration
        if "prompt_template" not in extra_info:
            try:
                extra_info["prompt_template"] = get_prompt_template(llm_prompt_template)
            except ValueError as e:
                if verbose:
                    print(f"[transform_example] Warning: {e}. Using default template.")
                extra_info["prompt_template"] = get_prompt_template("default")
        
        # Debug logging for LLM configuration
        if verbose:
            llm_config = {
                "problem": extra_info.get("problem"),
                "model": extra_info.get("model"),
                "temperature": extra_info.get("temperature"),
                "max_tokens": extra_info.get("max_tokens"),
                "timeout": extra_info.get("timeout"),
                "thinking_enabled": extra_info.get("thinking_enabled"),
                "thinking_budget": extra_info.get("thinking_budget"),
                "prompt_template": extra_info.get("prompt_template")
            }
            print(f"[transform_example] LLM configuration added to extra_info for idx={idx}:")
            print(json.dumps(llm_config, indent=2))

    # Add BLEURT specific configuration
    if match_type == "bleurt":
        extra_info["length_penalty"] = bleurt_length_penalty
        extra_info["length_threshold"] = bleurt_length_threshold
        extra_info["bleurt_checkpoint"] = bleurt_checkpoint
        if bleurt_device:
            extra_info["device"] = bleurt_device
        
        # Debug logging for BLEURT configuration
        if verbose:
            bleurt_config = {
                "length_penalty": extra_info.get("length_penalty"),
                "length_threshold": extra_info.get("length_threshold"), 
                "bleurt_checkpoint": extra_info.get("bleurt_checkpoint"),
                "device": extra_info.get("device")
            }
            print(f"[transform_example] BLEURT configuration added to extra_info for idx={idx}:")
            print(json.dumps(bleurt_config, indent=2))

    # Add Embedding specific configuration (shared between embedding and embedding_remote)
    if match_type in ["embedding", "embedding_remote"]:
        extra_info["length_penalty"] = embedding_length_penalty
        extra_info["length_threshold"] = embedding_length_threshold
        
        # Debug logging for Embedding configuration
        if verbose:
            embedding_config = {
                "length_penalty": extra_info.get("length_penalty"),
                "length_threshold": extra_info.get("length_threshold")
            }
            print(f"[transform_example] {match_type.title()} configuration added to extra_info for idx={idx}:")
            print(json.dumps(embedding_config, indent=2))

    # Decide on *data_source* according to requested matching type.
    if match_type == "lexical":
        data_source = "lexical_match_custom"
    elif match_type == "embedding":
        data_source = "embedding_match_custom"
    elif match_type == "embedding_remote":
        data_source = "embedding_remote_match_custom"
    elif match_type == "llm_judge":
        data_source = "llm_judge_custom"
    elif match_type == "bleurt":
        data_source = "bleurt_match_custom"
    else:
        raise ValueError(
            f"Unsupported match_type: {match_type!r}. Choose 'lexical', 'embedding', 'embedding_remote', 'llm_judge', or 'bleurt'."
        )

    # Build the verl record.
    record = {
        "data_source": data_source,
        # Only include the problem statement – the solution will be generated
        # by the model during RL fine-tuning.
        "prompt": [{"role": "user", "content": str(example["problem"]).strip()}],
        # Ability tag is arbitrary – we reuse "lexical_match" for compatibility.
        "ability": f"{match_type}_match",
        "reward_model": {
            "style": "model",
            # Ground-truth reference that the reward model will compare against.
            "ground_truth": str(example["solution"]).strip(),
        },
        "extra_info": extra_info,
    }

    # For quick sanity-checking: print the structure of the record for the first
    # few samples when *verbose* is enabled.
    if verbose and idx < 3:
        print("[transform_example] Preview of transformed record (truncated):")
        print(json.dumps(record, indent=2, ensure_ascii=False)[:1000])

    return record


def create_non_member_pairs(dataset, member_examples: List[dict], num_pairs: int, mode: str = "same_problem", seed: int = 42, verbose: bool = False) -> List[dict]:
    """
    Create non-member data by randomly pairing problems with solutions from the full dataset.
    
    Args:
        dataset: The full original dataset with problem-solution pairs
        member_examples: List of member examples (for same_problem mode)
        num_pairs: Number of non-member pairs to create (equal to member data size)
        mode: "same_problem" (use member problems with random solutions) or "full_random" (randomly select from full dataset)
        seed: Random seed for reproducible shuffling
        verbose: Whether to print debug information
        
    Returns:
        List of examples with mismatched problem-solution pairs
    """
    rng = random.Random(seed)
    
    if mode == "same_problem":
        # Mode 1: Use the same problems as members, but with random solutions
        if len(member_examples) != num_pairs:
            raise ValueError(f"Member examples count ({len(member_examples)}) must equal num_pairs ({num_pairs}) for same_problem mode")
        
        # Randomly sample solution indices from the full dataset
        solution_indices = rng.sample(range(len(dataset)), num_pairs)
        
        # Ensure no original pairs by fixing any matches
        for i in range(num_pairs):
            member_problem = member_examples[i]["problem"]
            # Find the original index of this member problem in the full dataset
            member_original_idx = None
            for j, ex in enumerate(dataset):
                if str(ex["problem"]).strip() == str(member_problem).strip():
                    member_original_idx = j
                    break
            
            # If we found the original index and it matches our solution index, find a different one
            if member_original_idx is not None and solution_indices[i] == member_original_idx:
                for j in range(len(dataset)):
                    if j != member_original_idx and j not in solution_indices:
                        solution_indices[i] = j
                        break
                else:
                    # If all indices are used, swap with another position
                    swap_idx = (i + 1) % num_pairs
                    if member_original_idx != solution_indices[swap_idx]:
                        solution_indices[i], solution_indices[swap_idx] = solution_indices[swap_idx], solution_indices[i]
        
        # Create non-member examples using member problems with random solutions
        non_member_examples = []
        for i in range(num_pairs):
            member_ex = member_examples[i]
            sol_idx = solution_indices[i]
            solution_ex = dataset[sol_idx]
            
            # Create new example with member problem but random solution
            non_member_ex = {
                "problem": str(member_ex["problem"]).strip(),
                "solution": str(solution_ex["solution"]).strip(),
                "answer": str(solution_ex.get("answer", "")).strip(),  # Use answer from solution source
                "subject": str(solution_ex.get("subject", "")).strip(),  # Use subject from solution source
                "level": solution_ex.get("level", 0),  # Use level from solution source
                "unique_id": str(member_ex.get("unique_id", "")).strip(),  # Keep member's unique_id
                "original_problem_idx": i,  # Index in member examples
                "original_solution_idx": sol_idx,  # Index in full dataset
                "is_member": False
            }
            
            non_member_examples.append(non_member_ex)
    
    elif mode == "full_random":
        # Mode 2: Randomly select from the full dataset (original behavior)
        # Randomly sample problem indices and solution indices separately
        problem_indices = rng.sample(range(len(dataset)), num_pairs)
        solution_indices = rng.sample(range(len(dataset)), num_pairs)
        
        # Ensure no original pairs by fixing any matches
        for i in range(num_pairs):
            if problem_indices[i] == solution_indices[i]:
                # Find a different solution index
                for j in range(len(dataset)):
                    if j != problem_indices[i] and j not in solution_indices:
                        solution_indices[i] = j
                        break
                else:
                    # If all indices are used, swap with another position
                    swap_idx = (i + 1) % num_pairs
                    if problem_indices[i] != solution_indices[swap_idx]:
                        solution_indices[i], solution_indices[swap_idx] = solution_indices[swap_idx], solution_indices[i]
        
        # Create non-member examples
        non_member_examples = []
        for i in range(num_pairs):
            prob_idx = problem_indices[i]
            sol_idx = solution_indices[i]
            
            # Take problem from one example, solution from another
            problem_ex = dataset[prob_idx]
            solution_ex = dataset[sol_idx]
            
            # Create new example with mismatched pair
            non_member_ex = {
                "problem": str(problem_ex["problem"]).strip(),
                "solution": str(solution_ex["solution"]).strip(),
                "answer": str(solution_ex.get("answer", "")).strip(),  # Use answer from solution source
                "subject": str(solution_ex.get("subject", "")).strip(),  # Use subject from solution source
                "level": solution_ex.get("level", 0),  # Use level from solution source
                "unique_id": str(problem_ex.get("unique_id", "")).strip(),  # Keep problem's unique_id
                "original_problem_idx": prob_idx,
                "original_solution_idx": sol_idx,
                "is_member": False
            }
            
            non_member_examples.append(non_member_ex)
    
    else:
        raise ValueError(f"Unknown random pairing mode: {mode}. Choose 'same_problem' or 'full_random'.")
    
    if verbose:
        print(f"[create_non_member_pairs] Created {len(non_member_examples)} non-member examples using mode '{mode}'")
        # Show a few examples of the mismatching
        for i in range(min(3, len(non_member_examples))):
            if mode == "same_problem":
                print(f"  Example {i}: member problem with solution from idx {solution_indices[i]}")
            else:
                print(f"  Example {i}: problem from idx {problem_indices[i]}, solution from idx {solution_indices[i]}")
    
    return non_member_examples


def subsample_member_indices(member_indices: List[int], target_size: int, seed: int = 42, verbose: bool = False) -> List[int]:
    """
    Subsample from the original member indices to get a smaller set of members.
    This is needed when we have more member examples than available non-member examples.
    
    Args:
        member_indices: Original list of member indices from fine-tuning
        target_size: Number of member indices to select
        seed: Random seed for reproducible selection
        verbose: Whether to print debug information
        
    Returns:
        List of selected member indices
    """
    if target_size >= len(member_indices):
        if verbose:
            print(f"[subsample_member_indices] WARNING: Requested size {target_size} >= available members {len(member_indices)}.")
            print(f"[subsample_member_indices] Using all {len(member_indices)} members.")
        return member_indices
    
    rng = random.Random(seed)
    selected_indices = rng.sample(member_indices, target_size)
    selected_indices.sort()  # For stability
    
    if verbose:
        print(f"[subsample_member_indices] Subsampled {target_size} indices from {len(member_indices)} original members")
        print(f"[subsample_member_indices] Selected indices: {selected_indices[:10]}..." if len(selected_indices) > 10 else f"[subsample_member_indices] Selected indices: {selected_indices}")
    
    return selected_indices


def create_non_member_from_unused(dataset, member_indices: List[int], num_pairs: int, seed: int = 42, verbose: bool = False) -> Tuple[List[dict], List[int]]:
    """
    Create non-member data by selecting examples from the dataset that were NOT used for fine-tuning.
    
    This is a safer approach than random pairing, as it uses real problem-solution pairs that are
    in-distribution but were not seen during training.
    
    Args:
        dataset: The full original dataset with problem-solution pairs
        member_indices: List of indices that were used as member data (fine-tuning data)
        num_pairs: Number of non-member pairs to create (should equal member data size)
        seed: Random seed for reproducible selection
        verbose: Whether to print debug information
        
    Returns:
        Tuple of:
        - List of non-member examples
        - List of member indices to use (may be subsampled from original if needed)
    """
    # Find all indices that were NOT used for member data
    all_indices = set(range(len(dataset)))
    member_indices_set = set(member_indices)
    unused_indices = list(all_indices - member_indices_set)
    
    # If we have fewer unused examples than requested, we need to:
    # 1. Use all unused examples as non-members
    # 2. Subsample from member indices to match this size
    if len(unused_indices) < num_pairs:
        if verbose:
            print(f"[create_non_member_from_unused] WARNING: Only {len(unused_indices)} unused examples available.")
            print(f"[create_non_member_from_unused] Will use all unused examples and subsample members to match.")
        
        # Use all unused examples
        selected_unused_indices = unused_indices
        selected_unused_indices.sort()  # For stability
        
        # Subsample from member indices to match the number of unused examples
        subsampled_member_indices = subsample_member_indices(
            member_indices=member_indices,
            target_size=len(unused_indices),
            seed=seed,  # Use same seed for reproducibility
            verbose=verbose
        )
    else:
        # Normal case: We have enough unused examples
        rng = random.Random(seed)
        selected_unused_indices = rng.sample(unused_indices, num_pairs)
        selected_unused_indices.sort()  # For stability
        subsampled_member_indices = member_indices  # Use all original members
    
    # Create non-member examples from unused data
    non_member_examples = []
    for i, idx in enumerate(selected_unused_indices):
        ex = dataset[idx]
        
        # Create non-member example (original problem-solution pair, just not seen during training)
        non_member_ex = {
            "problem": str(ex["problem"]).strip(),
            "solution": str(ex["solution"]).strip(),
            "answer": str(ex.get("answer", "")).strip(),
            "subject": str(ex.get("subject", "")).strip(),
            "level": ex.get("level", 0),
            "unique_id": str(ex.get("unique_id", "")).strip(),
            "original_idx": idx,
            "is_member": False
        }
        
        non_member_examples.append(non_member_ex)
    
    if verbose:
        print(f"[create_non_member_from_unused] Created {len(non_member_examples)} non-member examples from unused data")
        print(f"[create_non_member_from_unused] Total dataset size: {len(dataset)}, Member indices: {len(member_indices)}, Unused: {len(unused_indices)}")
        print(f"[create_non_member_from_unused] Selected unused indices: {selected_unused_indices[:10]}..." if len(selected_unused_indices) > 10 else f"[create_non_member_from_unused] Selected unused indices: {selected_unused_indices}")
    
    return non_member_examples, subsampled_member_indices


def add_member_flag(example, idx):
    """Add is_member=True flag to an example."""
    example_copy = dict(example)
    example_copy["is_member"] = True
    return example_copy


def update_target_gt_for_matching_problems(member_data, non_member_data, verbose: bool = False):
    """
    Update target_gt in extra_info for both member and non-member data when they have matching problems.
    
    When include_target_gt is True and we use same_problem mode, we want both member and non-member
    entries with the same problem to have a list containing both solutions in their target_gt.
    
    Args:
        member_data: List of member examples (already transformed)
        non_member_data: List of non-member examples (already transformed)
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (updated_member_data, updated_non_member_data)
    """
    # Create a mapping from problem text to (member_idx, non_member_idx) pairs
    problem_to_indices = {}
    
    # Map member problems
    for i, member_ex in enumerate(member_data):
        problem_text = str(member_ex["prompt"][0]["content"]).strip()
        if problem_text not in problem_to_indices:
            problem_to_indices[problem_text] = {"member": [], "non_member": []}
        problem_to_indices[problem_text]["member"].append(i)
    
    # Map non-member problems
    for i, non_member_ex in enumerate(non_member_data):
        problem_text = str(non_member_ex["prompt"][0]["content"]).strip()
        if problem_text not in problem_to_indices:
            problem_to_indices[problem_text] = {"member": [], "non_member": []}
        problem_to_indices[problem_text]["non_member"].append(i)
    
    # Update target_gt for matching problems
    updated_member_data = member_data.copy()
    updated_non_member_data = non_member_data.copy()
    
    matches_found = 0
    for problem_text, indices in problem_to_indices.items():
        member_indices = indices["member"]
        non_member_indices = indices["non_member"]
        
        if member_indices and non_member_indices:
            matches_found += 1
            if verbose:
                print(f"[update_target_gt] Found matching problem: {len(member_indices)} members, {len(non_member_indices)} non-members")
                print(f"  Problem: {problem_text[:100]}...")
            
            # Collect all solutions for this problem
            all_solutions = []
            
            # Add member solutions
            for member_idx in member_indices:
                member_ex = updated_member_data[member_idx]
                if "target_gt" in member_ex["extra_info"]:
                    solution = member_ex["extra_info"]["target_gt"]
                    if isinstance(solution, list):
                        all_solutions.extend(solution)
                    else:
                        all_solutions.append(solution)
            
            # Add non-member solutions
            for non_member_idx in non_member_indices:
                non_member_ex = updated_non_member_data[non_member_idx]
                if "target_gt" in non_member_ex["extra_info"]:
                    solution = non_member_ex["extra_info"]["target_gt"]
                    if isinstance(solution, list):
                        all_solutions.extend(solution)
                    else:
                        all_solutions.append(solution)
            
            # Remove duplicates while preserving order
            unique_solutions = []
            for sol in all_solutions:
                if sol not in unique_solutions:
                    unique_solutions.append(sol)
            
            # Update all matching entries with the combined target_gt
            for member_idx in member_indices:
                updated_member_data[member_idx]["extra_info"]["target_gt"] = unique_solutions.copy()
            
            for non_member_idx in non_member_indices:
                updated_non_member_data[non_member_idx]["extra_info"]["target_gt"] = unique_solutions.copy()
    
    if verbose:
        print(f"[update_target_gt] Updated target_gt for {matches_found} matching problems")
    
    return updated_member_data, updated_non_member_data


def _write_jsonl(path: str, rows: List[dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert HuggingFaceH4/MATH-500 dataset to verl RL parquet format "
            "with optional MIA (Membership Inference Attack) data generation."
        )
    )
    parser.add_argument(
        "--dataset_path",
        default="HuggingFaceH4/MATH-500",
        help="HuggingFace dataset path to load (default: HuggingFaceH4/MATH-500)",
    )
    parser.add_argument(
        "--dataset_split",
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--match_type",
        choices=["lexical", "embedding", "embedding_remote", "llm_judge", "bleurt"],
        default="lexical",
        help="Choose reward type: lexical (BM25/ratio/etc.), embedding (local FastText similarity), embedding_remote (remote TEI server similarity), llm_judge (LLM-as-a-judge evaluation), or bleurt (BLEURT-based evaluation).",
    )
    parser.add_argument(
        "--metric",
        default="bm25",
        help="Similarity metric to store in extra_info (e.g. bm25, ratio, levenshtein, embedding).",
    )
    parser.add_argument(
        "--include_target_gt",
        action="store_true",
        help=(
            "If set, include the ground-truth solution as 'target_gt' in extra_info so that "
            "reward functions can filter references."
        ),
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=500,
        help=(
            "Number of examples to use as member data (for MIA evaluation). "
            "Defaults to the full set (500). If greater than the dataset size, the full set is used. "
            "Note: When using MIA with unused_examples method, this should be <= the number of "
            "unused examples available after fine-tuning."
        ),
    )
    parser.add_argument(
        "--finetune_subset_size",
        type=int,
        default=None,
        help=(
            "Number of examples that were used for fine-tuning. This is used to determine which "
            "examples are 'unused' for non-member data. If not specified, assumes fine-tuning "
            "used the same subset_size as member data."
        ),
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="Random seed for deterministic subsampling.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/math500_match_custom_mia/rl",
        help="Directory where the output Parquet file and optional JSONL files will be saved.",
    )
    parser.add_argument(
        "--output_name",
        default="math500_mia",
        help="Base name used for MIA JSONL outputs when --mia is set.",
    )
    parser.add_argument(
        "--mia",
        action="store_true",
        help="If set, generate MIA data: members (original pairs) and nonmembers (mismatched pairs).",
    )
    parser.add_argument(
        "--mia_nonmember_method",
        choices=["random_pairing", "unused_examples"],
        default="unused_examples",
        help=(
            "Method for creating non-member data: "
            "'random_pairing' (original method: randomly pair problems with solutions), "
            "'unused_examples' (safer method: use examples not seen during fine-tuning)."
        ),
    )
    parser.add_argument(
        "--random_pairing_mode",
        choices=["same_problem", "full_random"],
        default="same_problem",
        help=(
            "Mode for random pairing method: "
            "'same_problem' (use member problems with random solutions), "
            "'full_random' (randomly select from full dataset)."
        ),
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to mirror the Parquet file to.",
    )
    parser.add_argument(
        "--llm_model",
        default="gemini/gemini-2.5-flash",
        help="LLM model to use for llm_judge reward type (default: gemini/gemini-2.5-flash).",
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=0.6,
        help="Temperature for LLM judge (default: 0.6 for balanced scoring).",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens for LLM judge response including thinking tokens (default: 4096).",
    )
    parser.add_argument(
        "--llm_timeout",
        type=int,
        default=30,
        help="Timeout for LLM API calls in seconds (default: 30).",
    )
    parser.add_argument(
        "--llm_prompt_template",
        default="default",
        help=f"LLM prompt template to use (default: default). Available: {', '.join(list_available_templates())}",
    )
    parser.add_argument(
        "--llm_thinking_enabled",
        action="store_true",
        default=True,
        help="Enable thinking mode for Gemini models for better reasoning (default: True).",
    )
    parser.add_argument(
        "--no_llm_thinking",
        action="store_true",
        help="Disable thinking mode for Gemini models (overrides --llm_thinking_enabled).",
    )
    parser.add_argument(
        "--llm_thinking_budget",
        type=int,
        default=2048,
        help="Number of tokens allocated for thinking in Gemini models (default: 2048).",
    )
    parser.add_argument(
        "--bleurt_checkpoint",
        default="lucadiliello/BLEURT-20",
        help="BLEURT model checkpoint to use (default: lucadiliello/BLEURT-20).",
    )
    parser.add_argument(
        "--bleurt_length_penalty",
        choices=["none", "ratio", "sqrt", "log", "quadratic", "exponential"],
        default="none",
        help=(
            "Length penalty type for BLEURT (default: none). Options:\n"
            "- none: No penalty\n"
            "- ratio: Linear penalty based on length ratio\n"
            "- sqrt: Square root of ratio (milder penalty)\n"
            "- log: Logarithmic penalty\n"
            "- quadratic: Quadratic penalty (ratio^2) for stronger penalization\n"
            "- exponential: Exponential penalty (e^(-ratio)) for aggressive penalization"
        ),
    )
    parser.add_argument(
        "--bleurt_length_threshold",
        type=float,
        default=1.5,
        help="Length threshold for applying penalty in BLEURT (default: 1.5).",
    )
    parser.add_argument(
        "--bleurt_device",
        default=None,
        help="Device to run BLEURT on ('cuda', 'cpu', or None for auto-detect).",
    )
    parser.add_argument(
        "--embedding_length_penalty",
        choices=["none", "ratio", "sqrt", "log", "quadratic", "exponential"],
        default="none",
        help=(
            "Length penalty type for embedding similarity (both local and remote) (default: none). Options:\n"
            "- none: No penalty\n"
            "- ratio: Linear penalty based on length ratio\n"
            "- sqrt: Square root of ratio (milder penalty)\n"
            "- log: Logarithmic penalty\n"
            "- quadratic: Quadratic penalty (ratio^2) for stronger penalization\n"
            "- exponential: Exponential penalty (e^(-ratio)) for aggressive penalization"
        ),
    )
    parser.add_argument(
        "--embedding_length_threshold",
        type=float,
        default=1.5,
        help="Length threshold for applying penalty in embedding similarity (both local and remote) (default: 1.5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging the preprocessing pipeline.",
    )

    args = parser.parse_args()

    # Handle thinking mode logic
    if args.no_llm_thinking:
        args.llm_thinking_enabled = False
    
    # Load dataset
    ds_full = datasets.load_dataset(args.dataset_path, split=args.dataset_split)
    
    if args.verbose:
        print(f"[main] Loaded {len(ds_full)} examples from {args.dataset_path} ({args.dataset_split} split)")

    # Determine the actual fine-tuning subset size
    finetune_size = args.finetune_subset_size if args.finetune_subset_size is not None else args.subset_size
    
    # Check if the requested subset size is valid for MIA
    total_dataset_size = len(ds_full)
    if args.mia and args.mia_nonmember_method == "unused_examples":
        unused_examples_available = total_dataset_size - finetune_size
        if args.subset_size > unused_examples_available:
            raise ValueError(
                f"For MIA with unused examples, member size ({args.subset_size}) cannot exceed "
                f"the number of unused examples ({unused_examples_available}). "
                f"Fine-tuning used {finetune_size} examples, leaving {unused_examples_available} unused. "
                f"Please reduce subset_size to {unused_examples_available} or less."
            )
    
    # Random subsampling for member data. If requested subset is >= len(ds_full), keep full set.
    if args.subset_size is not None and args.subset_size < len(ds_full):
        rng = random.Random(args.subset_seed)
        sampled_indices = rng.sample(range(len(ds_full)), args.subset_size)
        # Preserve original order for stability and easier debugging
        sampled_indices.sort()
        ds_members_subset = ds_full.select(sampled_indices)
        member_indices = sampled_indices  # Store for non-member generation
        if args.verbose:
            print(f"[main] Subsampled {len(ds_members_subset)} examples for member data using seed {args.subset_seed}")
            print(f"[main] Member indices: {member_indices[:10]}..." if len(member_indices) > 10 else f"[main] Member indices: {member_indices}")
            if args.mia and args.mia_nonmember_method == "unused_examples":
                print(f"[main] Will have {total_dataset_size - len(member_indices)} examples available for non-members")
    else:
        if args.mia and args.mia_nonmember_method == "unused_examples":
            raise ValueError(
                "Cannot use full dataset as members when MIA with unused_examples is enabled. "
                "This would leave no examples for non-members. Please specify a subset_size."
            )
        ds_members_subset = ds_full
        member_indices = list(range(len(ds_full)))  # All indices are members
        if args.verbose:
            print(f"[main] Using full dataset with {len(ds_members_subset)} examples for member data (no subsampling applied)")

    # Prepare transformation function with fixed parameters
    transform_fn = partial(
        transform_example,
        split=args.dataset_split,
        match_type=args.match_type,
        metric=args.metric,
        include_target_gt=args.include_target_gt,
        verbose=args.verbose,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_timeout=args.llm_timeout,
        llm_prompt_template=args.llm_prompt_template,
        llm_thinking_enabled=args.llm_thinking_enabled,
        llm_thinking_budget=args.llm_thinking_budget,
        bleurt_checkpoint=args.bleurt_checkpoint,
        bleurt_length_penalty=args.bleurt_length_penalty,
        bleurt_length_threshold=args.bleurt_length_threshold,
        bleurt_device=args.bleurt_device,
        embedding_length_penalty=args.embedding_length_penalty,
        embedding_length_threshold=args.embedding_length_threshold,
    )

    if args.mia:
        # MIA mode: Create both member and non-member data
        print("=== MIA MODE: Creating member and non-member datasets ===")
        print(f"Non-member generation method: {args.mia_nonmember_method}")
        
        # Create member data from subsampled dataset
        print(f"Creating member data from {len(ds_members_subset)} subsampled examples...")
        
        # Add is_member field to member examples before transformation
        ds_members_with_flag = ds_members_subset.map(add_member_flag, with_indices=True)
        ds_members = ds_members_with_flag.map(transform_fn, with_indices=True, remove_columns=ds_members_with_flag.column_names)
        
        # Create non-member data using selected method
        if args.mia_nonmember_method == "random_pairing":
            print(f"Creating {len(ds_members_subset)} non-member examples with random pairing mode '{args.random_pairing_mode}'...")
            
            # Convert member dataset to list of examples for same_problem mode
            member_examples_list = []
            for i in range(len(ds_members_subset)):
                ex = ds_members_subset[i]
                member_ex = {
                    "problem": str(ex["problem"]).strip(),
                    "solution": str(ex["solution"]).strip(),
                    "answer": str(ex.get("answer", "")).strip(),
                    "subject": str(ex.get("subject", "")).strip(),
                    "level": ex.get("level", 0),
                    "unique_id": str(ex.get("unique_id", "")).strip(),
                }
                member_examples_list.append(member_ex)
            
            non_member_examples = create_non_member_pairs(
                ds_full, 
                member_examples=member_examples_list,
                num_pairs=len(ds_members_subset), 
                mode=args.random_pairing_mode,
                seed=args.subset_seed + 1, 
                verbose=args.verbose
            )
            final_member_indices = member_indices  # No subsampling needed
        elif args.mia_nonmember_method == "unused_examples":
            print(f"Creating non-member examples from unused dataset entries (not seen during fine-tuning)...")
            
            #* Get the fine-tuning indices (what was actually used for training) *#
            finetune_rng = random.Random(args.subset_seed)
            finetune_indices = finetune_rng.sample(range(len(ds_full)), finetune_size)
            finetune_indices.sort()
            
            if args.verbose:
                print(f"[main] Fine-tuning used {len(finetune_indices)} examples: {finetune_indices[:10]}..." if len(finetune_indices) > 10 else f"[main] Fine-tuning indices: {finetune_indices}")
                print(f"[main] Member data uses {len(member_indices)} examples: {member_indices[:10]}..." if len(member_indices) > 10 else f"[main] Member indices: {member_indices}")
            
            non_member_examples, final_member_indices = create_non_member_from_unused(
                ds_full, finetune_indices, num_pairs=len(ds_members_subset), 
                seed=args.subset_seed + 1, verbose=args.verbose
            )
            
            # If we had to subsample members, update the member dataset
            if len(final_member_indices) < len(member_indices):
                print(f"Note: Subsampled member data from {len(member_indices)} to {len(final_member_indices)} examples to match available non-members")
                ds_members_subset = ds_full.select(final_member_indices)
        else:
            raise ValueError(f"Unknown non-member method: {args.mia_nonmember_method}")
        
        ds_non_members_raw = datasets.Dataset.from_list(non_member_examples)
        ds_non_members = ds_non_members_raw.map(transform_fn, with_indices=True, remove_columns=ds_non_members_raw.column_names)
        
        # Handle target_gt updates for matching problems if include_target_gt is True
        if args.include_target_gt and args.mia_nonmember_method == "random_pairing" and args.random_pairing_mode == "same_problem":
            print("Updating target_gt for matching problems between member and non-member data...")
            
            # Convert datasets to lists for processing
            member_data_list = [ds_members[i] for i in range(len(ds_members))]
            non_member_data_list = [ds_non_members[i] for i in range(len(ds_non_members))]
            
            # Update target_gt for matching problems
            updated_member_data, updated_non_member_data = update_target_gt_for_matching_problems(
                member_data_list, non_member_data_list, verbose=args.verbose
            )
            
            # Convert back to datasets
            ds_members = datasets.Dataset.from_list(updated_member_data)
            ds_non_members = datasets.Dataset.from_list(updated_non_member_data)
        
        # Combine member and non-member data for the final training set
        ds_combined = datasets.concatenate_datasets([ds_members, ds_non_members])
        
        # Shuffle the combined dataset
        ds_transformed = ds_combined.shuffle(seed=args.subset_seed)
        
        if args.verbose:
            print(f"[main] Created combined dataset: {len(ds_members)} members + {len(ds_non_members)} non-members = {len(ds_transformed)} total")
    else:
        # Standard mode: Just transform the subsampled dataset
        ds_transformed = ds_members_subset.map(transform_fn, with_indices=True, remove_columns=ds_members_subset.column_names)

    if args.verbose:
        print("\n[main] Finished transformation – preview of transformed record(s):")
        preview_recs = ds_transformed[:2]
        print(json.dumps(preview_recs, indent=2, ensure_ascii=False)[:1500])

    # Persist to local disk
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.parquet")
    ds_transformed.to_parquet(out_path)
    print(f"Parquet file saved at {out_path}")

    # Optionally copy to HDFS
    if args.hdfs_dir:
        hdfs_path = os.path.join(args.hdfs_dir, "train.parquet")
        makedirs(args.hdfs_dir)
        copy(out_path, hdfs_path)
        print(f"Also copied to HDFS: {hdfs_path}")

    # Generate MIA JSONL files if requested
    if args.mia:
        members_path = os.path.join(output_dir, f"{args.output_name}_members.jsonl")
        nonmembers_path = os.path.join(output_dir, f"{args.output_name}_nonmembers.jsonl")

        print(f"\n=== Generating MIA JSONL files ===")
        
        # Build member rows (original problem-solution pairs from subsampled data)
        members_rows = []
        for i in range(len(ds_members_subset)):
            ex = ds_members_subset[i]
            row = {
                "id": i,
                "problem": str(ex["problem"]).strip(),
                "solution": str(ex["solution"]).strip(),
                "answer": str(ex.get("answer", "")).strip(),
                "subject": str(ex.get("subject", "")).strip(),
                "level": ex.get("level", 0),
                "unique_id": str(ex.get("unique_id", "")).strip(),
                "is_member": True,
                "pair_type": "original"
            }
            members_rows.append(row)

        # Build non-member rows (either mismatched pairs or unused examples)
        nonmembers_rows = []
        
        for i, ex in enumerate(non_member_examples):
            row = {
                "id": i,
                "problem": str(ex["problem"]).strip(),
                "solution": str(ex["solution"]).strip(),
                "answer": str(ex.get("answer", "")).strip(),
                "subject": str(ex.get("subject", "")).strip(),
                "level": ex.get("level", 0),
                "unique_id": str(ex.get("unique_id", "")).strip(),
                "is_member": False,
            }
            
            # Add method-specific metadata
            if args.mia_nonmember_method == "random_pairing":
                row["pair_type"] = "mismatched"
                row["original_problem_idx"] = ex.get("original_problem_idx", -1)
                row["original_solution_idx"] = ex.get("original_solution_idx", -1)
            elif args.mia_nonmember_method == "unused_examples":
                row["pair_type"] = "unused_original"
                row["original_idx"] = ex.get("original_idx", -1)
            
            nonmembers_rows.append(row)

        _write_jsonl(members_path, members_rows)
        _write_jsonl(nonmembers_path, nonmembers_rows)
        
        print(f"Wrote MIA JSONL files:")
        print(f"  Members ({len(members_rows)} examples): {members_path}")
        print(f"  Non-members ({len(nonmembers_rows)} examples): {nonmembers_path}")
        
        # Verification
        if args.verbose:
            print(f"\n=== MIA Data Verification ===")
            print(f"Members sample:")
            for i in range(min(2, len(members_rows))):
                row = members_rows[i]
                print(f"  {i}: problem[:50]={row['problem'][:50]}...")
                print(f"      solution[:50]={row['solution'][:50]}...")
            
            print(f"Non-members sample:")
            for i in range(min(2, len(nonmembers_rows))):
                row = nonmembers_rows[i]
                print(f"  {i}: problem[:50]={row['problem'][:50]}...")
                print(f"      solution[:50]={row['solution'][:50]}...")
                if args.mia_nonmember_method == "random_pairing":
                    print(f"      (mismatched: problem from idx {row.get('original_problem_idx', -1)}, solution from idx {row.get('original_solution_idx', -1)})")
                elif args.mia_nonmember_method == "unused_examples":
                    print(f"      (unused original pair from idx {row.get('original_idx', -1)})")


if __name__ == "__main__":
    main()
