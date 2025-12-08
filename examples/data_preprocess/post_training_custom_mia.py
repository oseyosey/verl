"""
Convert post-training datasets (MATH-500, AIME, WildChat, etc.) to verl RL parquet format with MIA support.

This script:
- Loads datasets with problem-solution pairs OR messages format and converts to verl RL format
- Automatically normalizes datasets with 'messages' field to 'problem'/'solution' format
- Supports both lexical and embedding match types
- Optionally generates MIA (Membership Inference Attack) data with multiple methods:
  - Members: Original problem-solution pairs from the dataset (used for fine-tuning)
  - Non-members: Randomly paired, unused examples, perturbed solutions, or separate dataset
- Saves both RL parquet file and optional MIA JSONL files

Usage:
  # Basic conversion without MIA
  python post_training_custom_mia.py
  
  # With MIA data generation (safer unused examples method)
  python post_training_custom_mia.py --mia --mia_nonmember_method unused_examples
  
  # With MIA data generation (original random pairing method) - deduplicates when using --include_target_gt
  python post_training_custom_mia.py --mia --mia_nonmember_method random_pairing --random_pairing_mode full_random
  
  # With MIA data generation (same problem with random solutions) - deduplicates when using --include_target_gt
  python post_training_custom_mia.py --mia --mia_nonmember_method random_pairing --random_pairing_mode same_problem
  
  # With perturbed solutions (same problems as members) - deduplicates when using --include_target_gt
  python post_training_custom_mia.py --mia --mia_nonmember_method perturbed_solution --random_pairing_mode same_problem --perturbed_dataset_path YOUR_PERTURBED_DATASET
  
  # With perturbed solutions (random problems from full dataset) - deduplicates when using --include_target_gt
  python post_training_custom_mia.py --mia --mia_nonmember_method perturbed_solution --random_pairing_mode full_random --perturbed_dataset_path YOUR_PERTURBED_DATASET
  
  # Custom subset size and local embedding matching
  python post_training_custom_mia.py --subset_size 300 --match_type embedding --mia
  
  # Remote embedding matching (requires TEI server)
  python post_training_custom_mia.py --subset_size 300 --match_type embedding_remote --mia
  
  # LLM judge with custom prompt template and thinking mode
  python post_training_custom_mia.py --match_type llm_judge --llm_prompt_template detailed_rubric --subset_size 100
  
  # LLM judge without thinking mode (faster, lower cost)
  python post_training_custom_mia.py --match_type llm_judge --no_llm_thinking --llm_max_tokens 256 --subset_size 100
  
  # With assistant prefix to guide model generation (25% of solution)
  python post_training_custom_mia.py --enable_assistant_prefix --assistant_prefix_ratio 0.25
  
  # Reverse member and non-member labels (for testing MIA robustness)
  python post_training_custom_mia.py --mia --mia_nonmember_method perturbed_solution --random_pairing_mode same_problem --reverse_member
  
  # Target GT augmentation with random sampling (adds 1 additional solution to each target_gt)
  python post_training_custom_mia.py --mia --include_target_gt --augment_target_gt --augment_sampling_method random --augment_num_samples 1
  
  # Target GT augmentation with embedding similarity (top-2 most similar solutions)
  python post_training_custom_mia.py --mia --include_target_gt --augment_target_gt --augment_sampling_method embedding --augment_num_samples 2
  
  # Target GT augmentation with lexical similarity (top-3 most similar by Jaccard)
  python post_training_custom_mia.py --mia --include_target_gt --augment_target_gt --augment_sampling_method lexical --augment_num_samples 3
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
    from verl.utils.reward_score.llm_judge_prompts import get_prompt_template, list_available_templates
except ImportError:
    print("Cannot import llm_judge_prompts form verl")

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
    llm_batch_size: int = 128,
    # BLEURT-specific parameters
    bleurt_checkpoint: str = "lucadiliello/BLEURT-20",
    bleurt_length_penalty: str = "none",
    bleurt_length_threshold: float = 1.5,
    bleurt_device: str = None,
    # Embedding-specific parameters
    embedding_length_penalty: str = "none",
    embedding_length_threshold: float = 1.5,
    # Transformed ground truth
    transformed_solution: str = None,
    # Assistant prefix parameters
    enable_assistant_prefix: bool = False,
    assistant_prefix_ratio: float = 0.25,
    # Lexical metric parameters
    lexical_metric_profile: str = "default",
    lexical_custom_weights: List[float] = None,
    lexical_num_workers: int = 32,
    lexical_show_progress: bool = True,
    # Augmentation parameters
    augmented_solutions: List[str] = None,
    exclude_original_solution: bool = False,
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

    if match_type not in {"lexical", "embedding", "embedding_remote", "llm_judge", "llm_judge_remote", "bleurt"}:
        raise ValueError(
            f"Unsupported match_type: {match_type!r}. Choose 'lexical', 'embedding', 'embedding_remote', 'llm_judge', 'llm_judge_remote', or 'bleurt'."
        )

    # Construct *extra_info* section – accessible by the reward loader so that
    # it can forward *metric* to the scoring function.
    extra_info = {
        "split": split,
        "index": idx,
        "metric": metric,
    }
    
    # Add lexical-specific configuration
    if match_type == "lexical":
        # For lexical matching, use metric_profile instead of old metric system
        extra_info["metric_profile"] = lexical_metric_profile
        
        # Add custom weights if provided
        if lexical_custom_weights is not None:
            extra_info["custom_weights"] = lexical_custom_weights
        
        # Add number of workers for parallel processing
        extra_info["num_workers"] = lexical_num_workers
        
        # Add progress tracking flag
        extra_info["show_progress"] = lexical_show_progress
        
        # Map legacy metric names to new metric profiles for backward compatibility
        legacy_mapping = {
            "bm25": "default",
            "ratio": "default",
            "token_ratio": "default",
            "ordered_token": "default",
            "levenshtein": "default"
        }
        
        if metric in legacy_mapping and lexical_metric_profile == "default":
            # Override with legacy mapping only if user didn't explicitly set metric_profile
            extra_info["metric_profile"] = legacy_mapping[metric]
            if verbose and idx == 0:
                print(f"[transform_example] Mapped legacy metric '{metric}' to metric_profile '{legacy_mapping[metric]}'")
        
        if verbose and idx == 0:
            print(f"[transform_example] Lexical configuration: metric_profile='{extra_info['metric_profile']}', num_workers={lexical_num_workers}")
            if lexical_custom_weights:
                print(f"[transform_example] Using custom weights: {lexical_custom_weights}")
    
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
        base_solution = str(example["solution"]).strip()
        
        # Check if we have augmented solutions
        if augmented_solutions is not None and len(augmented_solutions) > 0:
            # Create list with or without original solution based on exclude_original_solution flag
            if exclude_original_solution:
                # Only use augmented solutions, exclude original
                extra_info["target_gt"] = augmented_solutions
                if verbose and idx == 0:
                    print(
                        f"[transform_example] Added augmented target_gt (EXCLUDING original) for first example:"
                    )
                    print(f"  - Original solution EXCLUDED")
                    print(f"  - Using {len(augmented_solutions)} augmented solution(s) only")
                    for i, aug_sol in enumerate(augmented_solutions[:2]):  # Show first 2
                        print(f"    [{i+1}] {aug_sol[:100]}...")
            else:
                # Include original solution + augmented solutions
                extra_info["target_gt"] = [base_solution] + augmented_solutions
                if verbose and idx == 0:
                    print(
                        f"[transform_example] Added augmented target_gt for first example:"
                    )
                    print(f"  - Original solution: {base_solution[:100]}...")
                    print(f"  - Added {len(augmented_solutions)} additional solution(s)")
                    for i, aug_sol in enumerate(augmented_solutions[:2]):  # Show first 2
                        print(f"    [{i+1}] {aug_sol[:100]}...")
        else:
            extra_info["target_gt"] = base_solution
            if verbose and idx == 0:
                print(
                    f"[transform_example] Added target_gt for first example: {base_solution[:100]}..."
                )

    # Preserve optional assistant prefix if upstream pipeline inserted one.
    if "assistant_prefix" in example:
        if verbose and idx == 0:
            print(
                f"[transform_example] Adding assistant_prefix for first example: {example['assistant_prefix']}"
            )
        extra_info["assistant_prefix"] = example["assistant_prefix"]
    
    # Generate assistant prefix from solution if enabled
    if enable_assistant_prefix and "assistant_prefix" not in extra_info:
        solution_text = str(example["solution"]).strip()
        words = solution_text.split()
        num_prefix_words = max(1, int(len(words) * assistant_prefix_ratio))
        assistant_prefix = " ".join(words[:num_prefix_words])
        extra_info["assistant_prefix"] = assistant_prefix
        
        if verbose and idx == 0:
            print(
                f"[transform_example] Generated assistant_prefix with {num_prefix_words}/{len(words)} words (ratio={assistant_prefix_ratio}): {assistant_prefix[:100]}..."
            )

    # Add transformed ground truth if provided
    if transformed_solution is not None:
        extra_info["transformed_ground_truth"] = str(transformed_solution).strip()
        if verbose and idx == 0:
            print(
                f"[transform_example] Added transformed_ground_truth for first example: {str(transformed_solution).strip()[:100]}..."
            )

    # Add LLM judge specific configuration (both local and remote)
    if match_type in ["llm_judge", "llm_judge_remote"]:
        # Add problem context for LLM judge prompt formatting
        extra_info["problem"] = str(example["problem"]).strip()
        
        # Add LLM configuration
        extra_info["model"] = llm_model
        extra_info["model_name"] = llm_model  # For compatibility with remote client
        extra_info["temperature"] = llm_temperature
        extra_info["max_tokens"] = llm_max_tokens
        extra_info["max_new_tokens"] = llm_max_tokens  # For compatibility with remote client
        extra_info["timeout"] = llm_timeout
        extra_info["thinking_enabled"] = llm_thinking_enabled
        extra_info["enable_thinking"] = llm_thinking_enabled  # For compatibility with remote client
        extra_info["thinking_budget"] = llm_thinking_budget
        extra_info["batch_size"] = llm_batch_size
        
        # Add prompt template name for LLM judge from configuration
        if "prompt_template" not in extra_info:
            try:
                # Validate template name exists, but store the name, not the full template
                get_prompt_template(llm_prompt_template)  # Just to validate it exists
                extra_info["prompt_template"] = llm_prompt_template
            except ValueError as e:
                if verbose:
                    print(f"[transform_example] Warning: {e}. Using default template.")
                extra_info["prompt_template"] = "default"
        
        # Debug logging for LLM configuration (only for first example)
        if verbose and idx == 0:
            llm_config = {
                "model": extra_info.get("model"),
                "model_name": extra_info.get("model_name"),
                "temperature": extra_info.get("temperature"),
                "max_tokens": extra_info.get("max_tokens"),
                "max_new_tokens": extra_info.get("max_new_tokens"),
                "timeout": extra_info.get("timeout"),
                "thinking_enabled": extra_info.get("thinking_enabled"),
                "enable_thinking": extra_info.get("enable_thinking"),
                "thinking_budget": extra_info.get("thinking_budget"),
                "batch_size": extra_info.get("batch_size"),
                "prompt_template": extra_info.get("prompt_template")
            }
            print(f"[transform_example] LLM Judge ({match_type}) configuration (applied to all examples):")
            print(json.dumps(llm_config, indent=2))

    # Add BLEURT specific configuration
    if match_type == "bleurt":
        extra_info["length_penalty"] = bleurt_length_penalty
        extra_info["length_threshold"] = bleurt_length_threshold
        extra_info["bleurt_checkpoint"] = bleurt_checkpoint
        if bleurt_device:
            extra_info["device"] = bleurt_device
        
        # Debug logging for BLEURT configuration (only for first example)
        if verbose and idx == 0:
            bleurt_config = {
                "length_penalty": extra_info.get("length_penalty"),
                "length_threshold": extra_info.get("length_threshold"), 
                "bleurt_checkpoint": extra_info.get("bleurt_checkpoint"),
                "device": extra_info.get("device")
            }
            print(f"[transform_example] BLEURT configuration (applied to all examples):")
            print(json.dumps(bleurt_config, indent=2))

    # Add Embedding specific configuration (shared between embedding and embedding_remote)
    if match_type in ["embedding", "embedding_remote"]:
        extra_info["length_penalty"] = embedding_length_penalty
        extra_info["length_threshold"] = embedding_length_threshold
        
        # Debug logging for Embedding configuration (only for first example)
        if verbose and idx == 0:
            embedding_config = {
                "length_penalty": extra_info.get("length_penalty"),
                "length_threshold": extra_info.get("length_threshold")
            }
            print(f"[transform_example] {match_type.title()} configuration (applied to all examples):")
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
    elif match_type == "llm_judge_remote":
        data_source = "llm_judge_remote_custom"
    elif match_type == "bleurt":
        data_source = "bleurt_match_custom"
    else:
        raise ValueError(
            f"Unsupported match_type: {match_type!r}. Choose 'lexical', 'embedding', 'embedding_remote', 'llm_judge', 'llm_judge_remote', or 'bleurt'."
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
    # sample when *verbose* is enabled.
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


def create_non_member_from_perturbed(perturbed_dataset, member_examples: List[dict], num_pairs: int, mode: str = "same_problem", seed: int = 42, verbose: bool = False, original_dataset=None, member_indices=None) -> List[dict]:
    """
    Create non-member data by using perturbed solutions from an external dataset.
    
    Args:
        perturbed_dataset: The perturbed dataset with same structure as original but different solutions
        member_examples: List of member examples (for same_problem mode)
        num_pairs: Number of non-member pairs to create (equal to member data size)
        mode: "same_problem" (use member problems with corresponding perturbed solutions) or "full_random" (randomly select problems from original dataset)
        seed: Random seed for reproducible shuffling
        verbose: Whether to print debug information
        original_dataset: The original dataset (required for full_random mode)
        member_indices: Original indices of member examples in the full dataset (required for same_problem mode)
        
    Returns:
        List of examples with perturbed problem-solution pairs
    """
    rng = random.Random(seed)
    
    if mode == "same_problem":
        # Mode 1: Use the same problems as members, but with corresponding perturbed solutions
        if len(member_examples) != num_pairs:
            raise ValueError(f"Member examples count ({len(member_examples)}) must equal num_pairs ({num_pairs}) for same_problem mode")
        
        if member_indices is None:
            raise ValueError("member_indices is required for same_problem mode to maintain correspondence")
        
        if len(member_indices) != num_pairs:
            raise ValueError(f"Member indices count ({len(member_indices)}) must equal num_pairs ({num_pairs}) for same_problem mode")
        
        # Create non-member examples using member problems with corresponding perturbed solutions
        non_member_examples = []
        for i in range(num_pairs):
            member_ex = member_examples[i]
            original_idx = member_indices[i]  # Use original index to get corresponding perturbed solution
            perturbed_ex = perturbed_dataset[original_idx]  # Use original index for correct correspondence
            
            # Create new example with member problem but perturbed solution
            non_member_ex = {
                "problem": str(member_ex["problem"]).strip(),
                "solution": str(perturbed_ex["solution"]).strip(),
                "answer": str(perturbed_ex.get("answer", "")).strip(),  # Use answer from perturbed source
                "subject": str(perturbed_ex.get("subject", "")).strip(),  # Use subject from perturbed source
                "level": perturbed_ex.get("level", 0),  # Use level from perturbed source
                "unique_id": str(member_ex.get("unique_id", "")).strip(),  # Keep member's unique_id
                "is_member": False,
            }
            
            non_member_examples.append(non_member_ex)
    
    elif mode == "full_random":
        # Mode 2: Randomly select problems from the original dataset, with corresponding perturbed solutions
        if original_dataset is None:
            raise ValueError("original_dataset is required for full_random mode")
        
        if len(original_dataset) != len(perturbed_dataset):
            raise ValueError(f"Original dataset size ({len(original_dataset)}) must match perturbed dataset size ({len(perturbed_dataset)})")
        
        # Randomly sample indices from the full original dataset
        random_indices = rng.sample(range(len(original_dataset)), min(num_pairs, len(original_dataset)))
        if len(original_dataset) < num_pairs:
            # If we need more pairs than available, sample with replacement
            additional_needed = num_pairs - len(original_dataset)
            random_indices.extend(rng.choices(range(len(original_dataset)), k=additional_needed))
        
        # Create non-member examples using random problems with their corresponding perturbed solutions
        non_member_examples = []
        for i in range(num_pairs):
            idx = random_indices[i]
            original_ex = original_dataset[idx]  # Problem from original dataset
            perturbed_ex = perturbed_dataset[idx]  # Corresponding perturbed solution
            
            # Use problem from original dataset, solution from perturbed dataset (maintaining i->i correspondence)
            non_member_ex = {
                "problem": str(original_ex["problem"]).strip(),  # Problem from original at random index
                "solution": str(perturbed_ex["solution"]).strip(),  # Corresponding perturbed solution
                "answer": str(perturbed_ex.get("answer", "")).strip(),  # Use answer from perturbed source
                "subject": str(perturbed_ex.get("subject", "")).strip(),  # Use subject from perturbed source
                "level": perturbed_ex.get("level", 0),  # Use level from perturbed source
                "unique_id": str(original_ex.get("unique_id", "")).strip(),  # Keep original's unique_id
                "is_member": False,
            }
            
            non_member_examples.append(non_member_ex)
    
    else:
        raise ValueError(f"Unknown perturbed pairing mode: {mode}. Choose 'same_problem' or 'full_random'.")
    
    if verbose:
        print(f"[create_non_member_from_perturbed] Created {len(non_member_examples)} non-member examples using mode '{mode}'")
        # Show a few examples of the pairing
        for i in range(min(3, len(non_member_examples))):
            if mode == "same_problem":
                if member_indices:
                    print(f"  Example {i}: member problem with perturbed solution from original idx {member_indices[i]}")
                else:
                    print(f"  Example {i}: member problem with corresponding perturbed solution")
            else:
                print(f"  Example {i}: random problem from original dataset with corresponding perturbed solution")
    
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
            # Also store separate member/non-member ground truths for MIA evaluation
            # IMPORTANT: Preserve MIA weights if present and store both member and non-member weights
            for member_idx in member_indices:
                member_ex = updated_member_data[member_idx]
                original_member_gt = member_ex["reward_model"]["ground_truth"]
                
                # Preserve existing MIA weight fields if they exist
                member_mia_weight = member_ex["extra_info"].get("mia_weight")
                mia_weight_tag = member_ex["extra_info"].get("mia_weight_tag")
                
                updated_member_data[member_idx]["extra_info"]["target_gt"] = unique_solutions.copy()
                # Store separate ground truths for MIA evaluation
                updated_member_data[member_idx]["extra_info"]["member_ground_truth"] = original_member_gt
                if len(unique_solutions) > 1:
                    # Find the non-member ground truth (the one that's not the member's)
                    nonmember_gt = unique_solutions[1] if unique_solutions[0] == original_member_gt else unique_solutions[0]
                    updated_member_data[member_idx]["extra_info"]["nonmember_ground_truth"] = nonmember_gt
                    updated_member_data[member_idx]["extra_info"]["has_nonmember_gt"] = True
                else:
                    updated_member_data[member_idx]["extra_info"]["has_nonmember_gt"] = False
                
                # Store both member and non-member MIA weights for later use
                if member_mia_weight is not None:
                    updated_member_data[member_idx]["extra_info"]["member_mia_weight"] = member_mia_weight
                if mia_weight_tag is not None:
                    updated_member_data[member_idx]["extra_info"]["mia_weight_tag"] = mia_weight_tag
            
            # Collect non-member MIA weights to store in member examples
            nonmember_mia_weights = []
            for non_member_idx in non_member_indices:
                non_member_ex = updated_non_member_data[non_member_idx]
                original_nonmember_gt = non_member_ex["reward_model"]["ground_truth"]
                
                # Preserve existing MIA weight fields if they exist
                nonmember_mia_weight = non_member_ex["extra_info"].get("mia_weight")
                mia_weight_tag = non_member_ex["extra_info"].get("mia_weight_tag")
                
                updated_non_member_data[non_member_idx]["extra_info"]["target_gt"] = unique_solutions.copy()
                # Store separate ground truths for MIA evaluation  
                updated_non_member_data[non_member_idx]["extra_info"]["nonmember_ground_truth"] = original_nonmember_gt
                if len(unique_solutions) > 1:
                    # Find the member ground truth (the one that's not the non-member's)
                    member_gt = unique_solutions[1] if unique_solutions[0] == original_nonmember_gt else unique_solutions[0]
                    updated_non_member_data[non_member_idx]["extra_info"]["member_ground_truth"] = member_gt
                    updated_non_member_data[non_member_idx]["extra_info"]["has_nonmember_gt"] = True
                else:
                    updated_non_member_data[non_member_idx]["extra_info"]["has_nonmember_gt"] = False
                
                # Restore MIA weights if they existed
                if nonmember_mia_weight is not None:
                    updated_non_member_data[non_member_idx]["extra_info"]["mia_weight"] = nonmember_mia_weight
                if mia_weight_tag is not None:
                    updated_non_member_data[non_member_idx]["extra_info"]["mia_weight_tag"] = mia_weight_tag
                
                # Collect non-member MIA weight for storage in member examples
                nonmember_mia_weights.append(nonmember_mia_weight)
            
            # Store non-member MIA weights in member examples (for cases where non-members get removed)
            if nonmember_mia_weights and len(nonmember_mia_weights) == len(member_indices):
                for i, member_idx in enumerate(member_indices):
                    if nonmember_mia_weights[i] is not None:
                        updated_member_data[member_idx]["extra_info"]["nonmember_mia_weight"] = nonmember_mia_weights[i]
    
    if verbose:
        if matches_found > 0:
            print(f"[update_target_gt] Updated target_gt for {matches_found} matching problems")
        else:
            print(f"[update_target_gt] No matching problems found between member and non-member data")
    
    return updated_member_data, updated_non_member_data


def remove_duplicate_problems(ds_members, ds_non_members, verbose: bool = False):
    """
    Remove non-member records that have the same problem text as member records.
    
    This is used in same_problem mode where we want to keep only the member record
    (with correct ground truth) and remove the redundant non-member record 
    (with incorrect ground truth) for the same problem.
    
    Args:
        ds_members: Dataset containing member records
        ds_non_members: Dataset containing non-member records
        verbose: Whether to print debug information
        
    Returns:
        Filtered non-member dataset with duplicates removed
    """
    # Collect all problem texts from member data
    member_problems = set()
    for i in range(len(ds_members)):
        member_ex = ds_members[i]
        problem_text = str(member_ex["prompt"][0]["content"]).strip()
        member_problems.add(problem_text)
    
    # Filter non-member data to exclude problems that exist in member data
    filtered_non_members = []
    duplicates_removed = 0
    
    for i in range(len(ds_non_members)):
        non_member_ex = ds_non_members[i]
        problem_text = str(non_member_ex["prompt"][0]["content"]).strip()
        
        if problem_text not in member_problems:
            # Keep non-member records with unique problems
            filtered_non_members.append(non_member_ex)
        else:
            # Skip non-member records with problems that exist in member data
            duplicates_removed += 1
    
    if verbose:
        if duplicates_removed > 0:
            print(f"[remove_duplicate_problems] Removed {duplicates_removed} duplicate non-member records")
            print(f"[remove_duplicate_problems] Kept {len(filtered_non_members)} unique non-member records")
        else:
            print(f"[remove_duplicate_problems] No duplicate problems found - kept all {len(filtered_non_members)} non-member records")
    
    # Convert back to dataset
    if filtered_non_members:
        return datasets.Dataset.from_list(filtered_non_members)
    else:
        # Return empty dataset with same structure if no non-members remain
        return datasets.Dataset.from_list([])


def _write_jsonl(path: str, rows: List[dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_and_normalize_mia_weights(
    members_path: str,
    nonmembers_path: str,
    verbose: bool = False
) -> Tuple[List[float], List[float]]:
    """
    Load MIA weights from JSONL files and normalize them together.
    
    Args:
        members_path: Path to members JSONL file with 'idx', 'id', and 'score' fields
        nonmembers_path: Path to non-members JSONL file with 'idx', 'id', and 'score' fields
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (normalized_member_weights, normalized_nonmember_weights)
        Both lists are in order of 'idx' field (0, 1, 2, ...)
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
    
    # Sort by idx to ensure correct order
    member_scores.sort(key=lambda x: x[0])
    nonmember_scores.sort(key=lambda x: x[0])
    
    # Extract scores
    member_weights = [score for _, score in member_scores]
    nonmember_weights = [score for _, score in nonmember_scores]
    
    # Combine all weights for normalization
    all_weights = np.array(member_weights + nonmember_weights)
    
    # Normalize to [0, 1] using min-max normalization
    min_weight = np.min(all_weights)
    max_weight = np.max(all_weights)
    
    if max_weight > min_weight:
        normalized_all = (all_weights - min_weight) / (max_weight - min_weight)
    else:
        # All weights are the same, set to 0.5
        normalized_all = np.full_like(all_weights, 0.5)
    
    # Split back into members and non-members
    normalized_member_weights = normalized_all[:len(member_weights)].tolist()
    normalized_nonmember_weights = normalized_all[len(member_weights):].tolist()
    
    if verbose:
        print(f"[load_and_normalize_mia_weights] Loaded {len(member_weights)} member weights and {len(nonmember_weights)} non-member weights")
        print(f"[load_and_normalize_mia_weights] Original range: [{min_weight:.4f}, {max_weight:.4f}]")
        print(f"[load_and_normalize_mia_weights] Normalized range: [0.0, 1.0]")
        print(f"[load_and_normalize_mia_weights] Member weights stats: mean={np.mean(normalized_member_weights):.4f}, std={np.std(normalized_member_weights):.4f}")
        print(f"[load_and_normalize_mia_weights] Non-member weights stats: mean={np.mean(normalized_nonmember_weights):.4f}, std={np.std(normalized_nonmember_weights):.4f}")
    
    return normalized_member_weights, normalized_nonmember_weights


def compute_lexical_similarities(query_solution: str, candidate_solutions: List[str]) -> List[float]:
    """Compute Jaccard similarity between query solution and candidates.
    
    Uses regex-based tokenizer by default to avoid subword overlap artifacts.
    Set env `DDRL_USE_TRANSFORMERS_TOKENIZER` to use transformers tokenizer backup.
    
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


# Global cache for embedding model to avoid reloading
_EMBEDDING_MODEL_CACHE = {}


def _get_embedding_model():
    """Get or load the embedding model (cached for efficiency).
    
    Returns:
        SentenceTransformer model instance
    """
    if "model" in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE["model"]
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as e:
        raise ImportError(f"sentence-transformers is required for embedding similarity: {e}")
    
    # Load Qwen3-8B model once and cache it
    model_name = "Qwen/Qwen3-Embedding-8B"
    
    try:
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
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
    
    # Cache the model
    _EMBEDDING_MODEL_CACHE["model"] = model
    return model


def _clear_embedding_model_cache():
    """Clear the embedding model cache and free GPU memory."""
    if "model" in _EMBEDDING_MODEL_CACHE:
        try:
            import torch
            import gc
            
            # Delete model
            del _EMBEDDING_MODEL_CACHE["model"]
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            print("✓ Cleared embedding model cache and freed GPU memory")
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")


def compute_embedding_similarities(query_solution: str, candidate_solutions: List[str]) -> List[float]:
    """Compute cosine similarity using Qwen3-8B embeddings.
    
    Uses a cached model instance to avoid reloading the model for every call.
    
    Args:
        query_solution: The query solution text
        candidate_solutions: List of candidate solution texts
        
    Returns:
        List of cosine similarity scores [0, 1]
    """
    import numpy as np
    
    # Get cached model (loads once, reuses thereafter)
    model = _get_embedding_model()
    
    # Encode all texts
    all_texts = [query_solution] + candidate_solutions
    embeddings = model.encode(
        all_texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    # Split embeddings
    query_emb = embeddings[0]
    cand_embs = embeddings[1:]
    
    # Compute cosine similarities (already normalized, so just dot product)
    similarities = cand_embs @ query_emb
    
    # Map from [-1, 1] to [0, 1]
    similarities = (similarities + 1.0) / 2.0
    
    return similarities.tolist()


def extract_year_from_url(url: str) -> int:
    """Extract year from AIME problem URL.
    
    Example: https://artofproblemsolving.com/wiki/index.php/2025_AIME_I_Problems/Problem_11
    Returns: 2025
    
    Args:
        url: URL string from the dataset
        
    Returns:
        Year as integer, or None if year cannot be extracted
    """
    import re
    match = re.search(r'/(\d{4})_AIME', url)
    if match:
        return int(match.group(1))
    return None


def normalize_dataset_format(dataset, verbose: bool = False):
    """Normalize dataset format by extracting problem/solution from messages if needed.
    
    If dataset has 'messages' field instead of 'problem'/'solution', extracts them.
    Otherwise, returns dataset as-is.
    
    Args:
        dataset: The dataset to normalize
        verbose: Whether to print debug information
        
    Returns:
        Normalized dataset with 'problem' and 'solution' fields
    """
    if len(dataset) == 0:
        return dataset
    
    # Check if dataset already has problem/solution fields
    if "problem" in dataset[0] and "solution" in dataset[0]:
        if verbose:
            print("[normalize_dataset_format] Dataset already has problem/solution fields, no normalization needed")
        return dataset
    
    # Check if dataset has messages field
    if "messages" not in dataset[0]:
        raise ValueError("Dataset must have either 'problem'/'solution' fields or 'messages' field")
    
    if verbose:
        print("[normalize_dataset_format] Converting messages format to problem/solution format")
    
    def extract_from_messages(example):
        """Extract problem and solution from messages field."""
        messages = example.get("messages", [])
        problem = ""
        solution = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if role == "user":
                problem = content
            elif role == "assistant":
                solution = content
        
        # Create new example with problem and solution fields
        new_example = dict(example)
        new_example["problem"] = problem
        new_example["solution"] = solution
        
        return new_example
    
    # Apply normalization to all examples
    normalized_dataset = dataset.map(extract_from_messages)
    
    if verbose:
        print(f"[normalize_dataset_format] Normalized {len(normalized_dataset)} examples")
        if len(normalized_dataset) > 0:
            sample = normalized_dataset[0]
            print(f"[normalize_dataset_format] Sample problem length: {len(sample.get('problem', ''))}")
            print(f"[normalize_dataset_format] Sample solution length: {len(sample.get('solution', ''))}")
    
    return normalized_dataset


def filter_dataset_by_years(dataset, years: List[int], verbose: bool = False):
    """Filter dataset to only include examples from specified years.
    
    Args:
        dataset: The dataset to filter (must have 'url' field)
        years: List of years to include
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (filtered_dataset, filtered_indices)
    """
    filtered_indices = []
    for idx in range(len(dataset)):
        if "url" in dataset[idx]:
            year = extract_year_from_url(dataset[idx]["url"])
            if year in years:
                filtered_indices.append(idx)
    
    if verbose:
        print(f"[filter_dataset_by_years] Filtered {len(filtered_indices)} examples from years {years} out of {len(dataset)} total")
    
    if len(filtered_indices) == 0:
        raise ValueError(f"No examples found for years {years}. Check that the dataset has 'url' field with year information.")
    
    return dataset.select(filtered_indices), filtered_indices


def filter_dataset_by_source(dataset, sources: List[str], verbose: bool = False):
    """Filter dataset to only include examples from specified sources.
    
    Args:
        dataset: The dataset to filter (must have 'source' field)
        sources: List of source names to include
        verbose: Whether to print debug information
        
    Returns:
        Tuple of (filtered_dataset, filtered_indices)
    """
    filtered_indices = []
    for idx in range(len(dataset)):
        if "source" in dataset[idx]:
            if dataset[idx]["source"] in sources:
                filtered_indices.append(idx)
    
    if verbose:
        print(f"[filter_dataset_by_source] Filtered {len(filtered_indices)} examples from sources {sources} out of {len(dataset)} total")
    
    if len(filtered_indices) == 0:
        raise ValueError(f"No examples found for sources {sources}. Check that the dataset has 'source' field. Available sources: {set(dataset['source'][:1000]) if len(dataset) > 0 and 'source' in dataset[0] else 'N/A'}")
    
    return dataset.select(filtered_indices), filtered_indices


def sample_additional_solutions(
    current_solution: str,
    current_idx: int,
    all_solutions: List[str],
    all_indices: List[Tuple],
    method: str,
    num_samples: int,
    seed: int = 42
) -> List[str]:
    """Sample additional solutions from the pool, excluding current example.
    
    Args:
        current_solution: Solution from the current example
        current_idx: Index of current example in the pool
        all_solutions: Pool of all solutions (from members + non-members)
        all_indices: Corresponding indices as tuples (type, idx)
        method: "random", "embedding", or "lexical"
        num_samples: Number of solutions to sample
        seed: Random seed for reproducibility
        
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
    
    # Check if we have enough candidates
    if len(candidate_solutions) == 0:
        return []
    
    num_samples = min(num_samples, len(candidate_solutions))
    
    if method == "random":
        # Random sampling
        rng = random.Random(seed + current_idx)  # Use different seed for each example
        sampled_indices = rng.sample(range(len(candidate_solutions)), num_samples)
        sampled_solutions = [candidate_solutions[i] for i in sampled_indices]
        
    elif method == "embedding":
        # Embedding-based sampling (top-k most similar)
        similarities = compute_embedding_similarities(current_solution, candidate_solutions)
        
        # Get indices of top-k most similar
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        top_k_indices = sorted_indices[:num_samples]
        sampled_solutions = [candidate_solutions[i] for i in top_k_indices]
        
    elif method == "lexical":
        # Lexical-based sampling (top-k most similar by Jaccard)
        similarities = compute_lexical_similarities(current_solution, candidate_solutions)
        
        # Get indices of top-k most similar
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        top_k_indices = sorted_indices[:num_samples]
        sampled_solutions = [candidate_solutions[i] for i in top_k_indices]
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return sampled_solutions


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
        "--transformed_dataset_path",
        default=None,
        help="Optional HuggingFace dataset path containing transformed ground truths (solutions)",
    )
    parser.add_argument(
        "--transformed_dataset_split",
        default="train",
        help="Dataset split to use for transformed dataset (default: train)",
    )
    parser.add_argument(
        "--perturbed_dataset_path",
        default=None,
        help="Optional HuggingFace dataset path containing perturbed solutions for MIA non-member generation",
    )
    parser.add_argument(
        "--perturbed_dataset_split",
        default="train",
        help="Dataset split to use for perturbed dataset (default: train)",
    )
    parser.add_argument(
        "--nonmember_dataset_path",
        default=None,
        help="Optional HuggingFace dataset path for non-member data when using separate_dataset method",
    )
    parser.add_argument(
        "--nonmember_dataset_split",
        default=None,
        help="Dataset split to use for non-member dataset (default: same as --dataset_split)",
    )
    parser.add_argument(
        "--nonmember_transformed_dataset_path",
        default=None,
        help="Optional HuggingFace dataset path containing transformed ground truths for non-member dataset",
    )
    parser.add_argument(
        "--nonmember_transformed_dataset_split",
        default=None,
        help="Dataset split to use for non-member transformed dataset (default: same as --nonmember_dataset_split)",
    )
    parser.add_argument(
        "--match_type",
        choices=["lexical", "embedding", "embedding_remote", "llm_judge", "llm_judge_remote", "bleurt"],
        default="lexical",
        help="Choose reward type: lexical (BM25/ratio/etc.), embedding (local FastText similarity), embedding_remote (remote TEI server similarity), llm_judge (LLM-as-a-judge evaluation), llm_judge_remote (remote vLLM server LLM-as-a-judge), or bleurt (BLEURT-based evaluation).",
    )
    parser.add_argument(
        "--metric",
        default="bm25",
        help="Similarity metric to store in extra_info (e.g. bm25, ratio, levenshtein, embedding). For lexical matching, this will be converted to appropriate metric_profile.",
    )
    parser.add_argument(
        "--lexical_metric_profile",
        default="default",
        help=(
            "Metric profile for lexical matching. Options: "
            "'default' (average of token_overlap, lcs_ratio_cand, ngram_coverage), "
            "'lexical_token_overlap', 'lexical_lcs_ratio', 'lexical_lcs_ratio_cand', "
            "'length_ratio', 'lexical_ngram_coverage', 'lexical_ngram_coverage_ref', "
            "'comprehensive' (all metrics weighted). "
            "Only used when --match_type is 'lexical'."
        ),
    )
    parser.add_argument(
        "--lexical_custom_weights",
        type=float,
        nargs="+",
        default=None,
        help="Custom weights for lexical metrics when using a metric profile. Number of weights must match the number of metrics in the selected profile.",
    )
    parser.add_argument(
        "--lexical_num_workers",
        type=int,
        default=32,
        help="Number of parallel workers for lexical metrics computation (default: 32). Set to 1 to disable parallelization.",
    )
    parser.add_argument(
        "--lexical_show_progress",
        action="store_true",
        help="Show progress bar and throughput metrics during lexical reward computation (useful for monitoring performance during training).",
    )
    parser.add_argument(
        "--include_target_gt",
        action="store_true",
        help=(
            "If set, include the ground-truth solution as 'target_gt' in extra_info so that "
            "reward functions can filter references. This also removes duplicate non-member "
            "records that have the same problems as members to avoid storing redundant data."
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
        choices=["random_pairing", "unused_examples", "perturbed_solution", "separate_dataset"],
        default="unused_examples",
        help=(
            "Method for creating non-member data: "
            "'random_pairing' (original method: randomly pair problems with solutions), "
            "'unused_examples' (safer method: use examples not seen during fine-tuning), "
            "'perturbed_solution' (use perturbed solutions from external dataset), "
            "'separate_dataset' (use a completely separate dataset for non-members)."
        ),
    )
    parser.add_argument(
        "--random_pairing_mode",
        choices=["same_problem", "full_random"],
        default="same_problem",
        help=(
            "Mode for random pairing and perturbed solution methods: "
            "'same_problem' (use member problems with random/perturbed solutions), "
            "'full_random' (randomly select problems from full dataset with corresponding solutions)."
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
        "--llm_batch_size",
        type=int,
        default=128,
        help="Batch size for LLM judge remote processing (default: 1).",
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
        "--enable_assistant_prefix",
        action="store_true",
        help="Enable assistant prefix generation from solution text to guide model generation.",
    )
    parser.add_argument(
        "--assistant_prefix_ratio",
        type=float,
        default=0.25,
        help="Ratio of words from solution to use as assistant prefix (default: 0.25 for 25%%).",
    )
    parser.add_argument(
        "--reverse_member",
        action="store_true",
        help=(
            "If set, reverse the member and non-member labels. Original members become non-members "
            "and vice versa. This is useful for testing MIA robustness by flipping the training signal."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging the preprocessing pipeline.",
    )
    parser.add_argument(
        "--mia_weights_members",
        default=None,
        help="Path to JSONL file containing MIA weights for member examples (e.g., min_k++_members.jsonl).",
    )
    parser.add_argument(
        "--mia_weights_nonmembers",
        default=None,
        help="Path to JSONL file containing MIA weights for non-member examples (e.g., min_k++_nonmembers.jsonl).",
    )
    parser.add_argument(
        "--mia_weights_tag",
        choices=["loss", "loss_ref", "min_k", "min_k++"],
        default=None,
        help="Tag identifying the type of MIA weights (loss, loss_ref, min_k, or min_k++). Required when MIA weight files are provided.",
    )
    parser.add_argument(
        "--augment_target_gt",
        action="store_true",
        help="Augment target_gt with additional solutions from other examples in the MIA subset (requires --mia and --include_target_gt).",
    )
    parser.add_argument(
        "--augment_sampling_method",
        choices=["random", "embedding", "lexical"],
        default="random",
        help="Method for sampling additional solutions: random, embedding (Qwen3-8B similarity), or lexical (Jaccard similarity) (default: random).",
    )
    parser.add_argument(
        "--augment_num_samples",
        type=int,
        default=1,
        help="Number of additional solutions to sample and add to target_gt (default: 1).",
    )
    parser.add_argument(
        "--filter_by_year",
        action="store_true",
        help="Enable year-based filtering for member and non-member data (requires dataset with 'url' field containing year information).",
    )
    parser.add_argument(
        "--nonmember_years",
        type=int,
        nargs="+",
        default=[2025],
        help="List of years to use for non-member data (default: [2025]). Only used when --filter_by_year is enabled.",
    )
    parser.add_argument(
        "--member_years",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024],
        help="List of years to use for member data (default: [2021, 2022, 2023, 2024]). Only used when --filter_by_year is enabled.",
    )
    parser.add_argument(
        "--source_filter",
        type=str,
        nargs="+",
        default=None,
        help="Filter dataset by specific source(s) (e.g., ai2-adapt-dev/numinamath_tir_math_decontaminated). Requires dataset with 'source' field.",
    )
    parser.add_argument(
        "--remove_nonmember_original_gt",
        action="store_true",
        help="Remove original ground truth from non-members, keeping only augmented solutions (requires --augment_target_gt).",
    )

    args = parser.parse_args()

    # Handle thinking mode logic
    if args.no_llm_thinking:
        args.llm_thinking_enabled = False
    
    # Validate year filtering arguments
    if args.filter_by_year:
        if not args.mia:
            raise ValueError("--filter_by_year requires --mia to be enabled")
        if len(args.nonmember_years) == 0:
            raise ValueError("--nonmember_years must contain at least one year")
        if len(args.member_years) == 0:
            raise ValueError("--member_years must contain at least one year")
        # Check for overlap
        overlap = set(args.member_years) & set(args.nonmember_years)
        if overlap:
            print(f"WARNING: Years {overlap} appear in both member_years and nonmember_years")
    
    # Validate augmentation arguments
    if args.augment_target_gt:
        if not args.mia:
            raise ValueError("--augment_target_gt requires --mia to be enabled")
        if not args.include_target_gt:
            raise ValueError("--augment_target_gt requires --include_target_gt to be enabled")
        if args.augment_num_samples < 1:
            raise ValueError("--augment_num_samples must be at least 1")
    
    # Validate remove_nonmember_original_gt argument
    if args.remove_nonmember_original_gt:
        if not args.augment_target_gt:
            raise ValueError("--remove_nonmember_original_gt requires --augment_target_gt to be enabled")
    
    # Validate MIA weights arguments
    if args.mia_weights_members or args.mia_weights_nonmembers:
        if not (args.mia_weights_members and args.mia_weights_nonmembers):
            raise ValueError("Both --mia_weights_members and --mia_weights_nonmembers must be provided together")
        if not args.mia_weights_tag:
            raise ValueError("--mia_weights_tag is required when MIA weight files are provided")
        if not args.mia:
            raise ValueError("--mia flag must be enabled when using MIA weights")
        
        # Check that files exist
        if not os.path.exists(args.mia_weights_members):
            raise FileNotFoundError(f"MIA weights file not found: {args.mia_weights_members}")
        if not os.path.exists(args.mia_weights_nonmembers):
            raise FileNotFoundError(f"MIA weights file not found: {args.mia_weights_nonmembers}")
    
    # Validate separate_dataset method arguments
    if args.mia and args.mia_nonmember_method == "separate_dataset":
        if not args.nonmember_dataset_path:
            raise ValueError("--nonmember_dataset_path is required when using --mia_nonmember_method separate_dataset")
        # Set default split if not provided
        if args.nonmember_dataset_split is None:
            args.nonmember_dataset_split = args.dataset_split
        # Set default transformed split if not provided
        if args.nonmember_transformed_dataset_path and args.nonmember_transformed_dataset_split is None:
            args.nonmember_transformed_dataset_split = args.nonmember_dataset_split
    
    # Load dataset
    ds_full = datasets.load_dataset(args.dataset_path, split=args.dataset_split)
    
    if args.verbose:
        print(f"[main] Loaded {len(ds_full)} examples from {args.dataset_path} ({args.dataset_split} split)")
    
    # Normalize dataset format (convert messages to problem/solution if needed)
    ds_full = normalize_dataset_format(ds_full, verbose=args.verbose)
    
    # Apply source filtering if enabled
    if args.source_filter is not None and len(args.source_filter) > 0:
        print("\n=== Source-based Filtering Enabled ===")
        print(f"Source filter: {args.source_filter}")
        
        # Check if dataset has 'source' field
        if len(ds_full) > 0 and "source" not in ds_full[0]:
            raise ValueError("--source_filter requires dataset to have 'source' field, but it was not found")
        
        ds_full, source_filtered_indices = filter_dataset_by_source(
            ds_full, args.source_filter, verbose=args.verbose
        )
        
        if args.verbose:
            print(f"[main] Source filtering: {len(ds_full)} examples from sources {args.source_filter}")
    
    # Apply year-based filtering if enabled
    member_pool_indices = None
    nonmember_pool_indices = None
    
    if args.filter_by_year:
        print("\n=== Year-based Filtering Enabled ===")
        print(f"Member years: {args.member_years}")
        print(f"Non-member years: {args.nonmember_years}")
        
        # Check if dataset has 'url' field
        if len(ds_full) > 0 and "url" not in ds_full[0]:
            raise ValueError("--filter_by_year requires dataset to have 'url' field, but it was not found")
        
        # Filter for member pool
        ds_member_pool, member_pool_indices = filter_dataset_by_years(
            ds_full, args.member_years, verbose=args.verbose
        )
        
        # Filter for non-member pool
        ds_nonmember_pool, nonmember_pool_indices = filter_dataset_by_years(
            ds_full, args.nonmember_years, verbose=args.verbose
        )
        
        # Validate subset size against filtered pools
        if args.subset_size > len(ds_member_pool):
            raise ValueError(
                f"subset_size ({args.subset_size}) exceeds available member data after year filtering "
                f"({len(ds_member_pool)} examples from years {args.member_years}). "
                f"Please reduce subset_size to {len(ds_member_pool)} or less."
            )
        
        if args.mia and args.mia_nonmember_method != "unused_examples":
            # For random_pairing and perturbed_solution, we sample from the non-member pool
            if args.subset_size > len(ds_nonmember_pool):
                raise ValueError(
                    f"subset_size ({args.subset_size}) exceeds available non-member data after year filtering "
                    f"({len(ds_nonmember_pool)} examples from years {args.nonmember_years}). "
                    f"Please reduce subset_size to {len(ds_nonmember_pool)} or less."
                )
        
        if args.verbose:
            print(f"[main] Year filtering results:")
            print(f"  - Member pool: {len(ds_member_pool)} examples from years {args.member_years}")
            print(f"  - Non-member pool: {len(ds_nonmember_pool)} examples from years {args.nonmember_years}")
    
    # Load transformed dataset if provided
    ds_transformed = None
    if args.transformed_dataset_path:
        ds_transformed = datasets.load_dataset(args.transformed_dataset_path, split=args.transformed_dataset_split)
        if args.verbose:
            print(f"[main] Loaded {len(ds_transformed)} transformed examples from {args.transformed_dataset_path} ({args.transformed_dataset_split} split)")
        # Normalize transformed dataset format
        ds_transformed = normalize_dataset_format(ds_transformed, verbose=args.verbose)
        if len(ds_transformed) != len(ds_full):
            raise ValueError(
                f"Transformed dataset size ({len(ds_transformed)}) does not match main dataset size ({len(ds_full)})"
            )
    
    # Load separate non-member dataset if using separate_dataset method
    ds_nonmember_full = None
    ds_nonmember_transformed = None
    if args.mia and args.mia_nonmember_method == "separate_dataset":
        ds_nonmember_full = datasets.load_dataset(args.nonmember_dataset_path, split=args.nonmember_dataset_split)
        if args.verbose:
            print(f"[main] Loaded {len(ds_nonmember_full)} non-member examples from {args.nonmember_dataset_path} ({args.nonmember_dataset_split} split)")
        
        # Normalize non-member dataset format
        ds_nonmember_full = normalize_dataset_format(ds_nonmember_full, verbose=args.verbose)
        
        # Validate required fields (after normalization)
        if len(ds_nonmember_full) > 0:
            if "problem" not in ds_nonmember_full[0] or "solution" not in ds_nonmember_full[0]:
                raise ValueError("Non-member dataset must have 'problem' and 'solution' fields after normalization")
        
        # Load non-member transformed dataset if provided
        if args.nonmember_transformed_dataset_path:
            ds_nonmember_transformed = datasets.load_dataset(args.nonmember_transformed_dataset_path, split=args.nonmember_transformed_dataset_split)
            if args.verbose:
                print(f"[main] Loaded {len(ds_nonmember_transformed)} transformed non-member examples from {args.nonmember_transformed_dataset_path} ({args.nonmember_transformed_dataset_split} split)")
            # Normalize non-member transformed dataset format
            ds_nonmember_transformed = normalize_dataset_format(ds_nonmember_transformed, verbose=args.verbose)
            if len(ds_nonmember_transformed) != len(ds_nonmember_full):
                raise ValueError(
                    f"Non-member transformed dataset size ({len(ds_nonmember_transformed)}) does not match non-member dataset size ({len(ds_nonmember_full)})"
                )
    
    # Load perturbed dataset if provided and using perturbed_solution method
    ds_perturbed = None
    if args.mia and args.mia_nonmember_method == "perturbed_solution":
        if not args.perturbed_dataset_path:
            raise ValueError("perturbed_dataset_path is required when using mia_nonmember_method='perturbed_solution'")
        ds_perturbed = datasets.load_dataset(args.perturbed_dataset_path, split=args.perturbed_dataset_split)
        if args.verbose:
            print(f"[main] Loaded {len(ds_perturbed)} perturbed examples from {args.perturbed_dataset_path} ({args.perturbed_dataset_split} split)")
        # Normalize perturbed dataset format
        ds_perturbed = normalize_dataset_format(ds_perturbed, verbose=args.verbose)
        if len(ds_perturbed) != len(ds_full):
            raise ValueError(
                f"Perturbed dataset size ({len(ds_perturbed)}) does not match main dataset size ({len(ds_full)})"
            )

    # Determine the actual fine-tuning subset size
    finetune_size = args.finetune_subset_size if args.finetune_subset_size is not None else args.subset_size
    
    # Check if the requested subset size is valid for MIA
    total_dataset_size = len(ds_full)
    if args.mia and args.mia_nonmember_method == "separate_dataset":
        if args.subset_size > len(ds_nonmember_full):
            raise ValueError(
                f"For MIA with separate_dataset, member size ({args.subset_size}) cannot exceed "
                f"the number of available non-member examples ({len(ds_nonmember_full)}). "
                f"Please reduce subset_size to {len(ds_nonmember_full)} or less."
            )
    if args.mia and args.mia_nonmember_method == "unused_examples":
        if args.filter_by_year:
            # With year filtering, we need to check if non-member pool has enough examples
            # In unused_examples mode, non-members come from the non-member pool
            unused_examples_available = len(ds_nonmember_pool)
            if args.subset_size > unused_examples_available:
                raise ValueError(
                    f"For MIA with unused examples and year filtering, member size ({args.subset_size}) cannot exceed "
                    f"the number of available non-member examples ({unused_examples_available} from years {args.nonmember_years}). "
                    f"Please reduce subset_size to {unused_examples_available} or less."
                )
        else:
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
        
        if args.filter_by_year:
            # Sample from the filtered member pool
            sampled_indices_in_pool = rng.sample(range(len(member_pool_indices)), args.subset_size)
            sampled_indices_in_pool.sort()
            # Map back to original dataset indices
            sampled_indices = [member_pool_indices[i] for i in sampled_indices_in_pool]
            sampled_indices.sort()
            ds_members_subset = ds_full.select(sampled_indices)
            member_indices = sampled_indices
            if args.verbose:
                print(f"[main] Subsampled {len(ds_members_subset)} examples for member data from year-filtered pool using seed {args.subset_seed}")
                print(f"[main] Member indices: {member_indices[:10]}..." if len(member_indices) > 10 else f"[main] Member indices: {member_indices}")
                if args.mia and args.mia_nonmember_method == "unused_examples":
                    print(f"[main] Will have {len(nonmember_pool_indices)} examples available for non-members from years {args.nonmember_years}")
        else:
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
    # We need to track the original indices when using subsampled data
    idx_to_original = {}
    if args.subset_size is not None and args.subset_size < len(ds_full):
        for new_idx, orig_idx in enumerate(sampled_indices):
            idx_to_original[new_idx] = orig_idx
    
    def transform_with_transformed(example, idx, augmented_solutions=None, exclude_original_solution=False):
        transformed_sol = None
        # Check if this is a non-member from separate dataset
        is_separate_dataset_nonmember = example.get("from_separate_dataset", False)
        
        if is_separate_dataset_nonmember:
            # For non-members from separate dataset, use non-member transformed dataset if available
            if ds_nonmember_transformed is not None:
                orig_idx = example.get("original_idx", idx)
                if orig_idx < len(ds_nonmember_transformed):
                    transformed_sol = ds_nonmember_transformed[orig_idx]["solution"]
            # Otherwise, use original solution (no transformation)
        elif ds_transformed is not None:
            # Get the original index - check multiple sources:
            # 1. If example has 'original_idx' field (for non-members from unused_examples)
            # 2. If example has 'original_solution_idx' field (for non-members from random_pairing/perturbed)
            # 3. Otherwise use idx_to_original mapping (for members)
            orig_idx = None
            if "original_idx" in example:
                # Non-member from unused_examples - use stored original index
                orig_idx = example["original_idx"]
            elif "original_solution_idx" in example:
                # Non-member from random_pairing/perturbed - use solution's original index
                orig_idx = example["original_solution_idx"]
            else:
                # Member data - use idx_to_original mapping
                orig_idx = idx_to_original.get(idx, idx)
            
            if orig_idx < len(ds_transformed):
                transformed_sol = ds_transformed[orig_idx]["solution"]
        
        return transform_example(
            example=example,
            idx=idx,
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
            llm_batch_size=args.llm_batch_size,
            bleurt_checkpoint=args.bleurt_checkpoint,
            bleurt_length_penalty=args.bleurt_length_penalty,
            bleurt_length_threshold=args.bleurt_length_threshold,
            bleurt_device=args.bleurt_device,
            embedding_length_penalty=args.embedding_length_penalty,
            embedding_length_threshold=args.embedding_length_threshold,
            transformed_solution=transformed_sol,
            enable_assistant_prefix=args.enable_assistant_prefix,
            assistant_prefix_ratio=args.assistant_prefix_ratio,
            lexical_metric_profile=args.lexical_metric_profile,
            lexical_custom_weights=args.lexical_custom_weights,
            lexical_num_workers=args.lexical_num_workers,
            lexical_show_progress=args.lexical_show_progress,
            augmented_solutions=augmented_solutions,
            exclude_original_solution=exclude_original_solution,
        )
    
    transform_fn = transform_with_transformed

    if args.mia:
        # MIA mode: Create both member and non-member data
        print("=== MIA MODE: Creating member and non-member datasets ===")
        print(f"Non-member generation method: {args.mia_nonmember_method}")
        
        # Create member data from subsampled dataset
        print(f"Creating member data from {len(ds_members_subset)} examples...")
        
        # Add is_member field to member examples before transformation
        ds_members_with_flag = ds_members_subset.map(add_member_flag, with_indices=True)
        
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
            
            # Use filtered non-member pool if year filtering is enabled
            if args.filter_by_year:
                nonmember_dataset = ds_full.select(nonmember_pool_indices)
                if args.verbose:
                    print(f"[main] Using year-filtered non-member pool: {len(nonmember_dataset)} examples from years {args.nonmember_years}")
            else:
                nonmember_dataset = ds_full
            
            non_member_examples = create_non_member_pairs(
                nonmember_dataset, 
                member_examples=member_examples_list,
                num_pairs=len(ds_members_subset), 
                mode=args.random_pairing_mode,
                seed=args.subset_seed + 1, 
                verbose=args.verbose
            )
            final_member_indices = member_indices  # No subsampling needed
        elif args.mia_nonmember_method == "unused_examples":
            print(f"Creating non-member examples from unused dataset entries...")
            
            if args.filter_by_year:
                # With year filtering, non-members come from the non-member pool
                # Members were already sampled from member pool
                # Non-members should be unused examples from the non-member pool
                if args.verbose:
                    print(f"[main] Using year-filtered non-member pool: {len(nonmember_pool_indices)} examples from years {args.nonmember_years}")
                
                # For unused_examples with year filtering:
                # - Members come from member_years (already sampled)
                # - Non-members come from nonmember_years (the entire pool, since they're unused by definition)
                # Sample from non-member pool
                nonmember_rng = random.Random(args.subset_seed + 1)
                if len(nonmember_pool_indices) >= len(ds_members_subset):
                    selected_nonmember_indices_in_pool = nonmember_rng.sample(range(len(nonmember_pool_indices)), len(ds_members_subset))
                    selected_nonmember_indices_in_pool.sort()
                    selected_nonmember_indices = [nonmember_pool_indices[i] for i in selected_nonmember_indices_in_pool]
                else:
                    # Use all non-member pool indices
                    selected_nonmember_indices = nonmember_pool_indices
                
                selected_nonmember_indices.sort()
                
                # Create non-member examples from selected indices
                non_member_examples = []
                for idx in selected_nonmember_indices:
                    ex = ds_full[idx]
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
                
                final_member_indices = member_indices
                
                if args.verbose:
                    print(f"[main] Created {len(non_member_examples)} non-member examples from year-filtered pool")
                    print(f"[main] Non-member indices: {selected_nonmember_indices[:10]}..." if len(selected_nonmember_indices) > 10 else f"[main] Non-member indices: {selected_nonmember_indices}")
            else:
                #* Get the fine-tuning indices (what was actually used for training) *#
                finetune_rng = random.Random(args.subset_seed)
                finetune_indices = finetune_rng.sample(range(len(ds_full)), finetune_size)
                finetune_indices.sort()
                
                if args.verbose:
                    print(f"[main] Fine-tuning used {len(finetune_indices)} examples, {len(ds_full) - len(finetune_indices)} unused")
                
                non_member_examples, final_member_indices = create_non_member_from_unused(
                    ds_full, finetune_indices, num_pairs=len(ds_members_subset), 
                    seed=args.subset_seed + 1, verbose=args.verbose
                )
                
                # If we had to subsample members, update the member dataset
                if len(final_member_indices) < len(member_indices):
                    print(f"Note: Subsampled member data from {len(member_indices)} to {len(final_member_indices)} examples to match available non-members")
                    ds_members_subset = ds_full.select(final_member_indices)
        elif args.mia_nonmember_method == "perturbed_solution":
            print(f"Creating {len(ds_members_subset)} non-member examples with perturbed solutions mode '{args.random_pairing_mode}'...")
            
            # Convert member dataset to list of examples for perturbed solution method
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
            
            # For perturbed_solution, we need to use the filtered pools if year filtering is enabled
            if args.filter_by_year:
                # Use filtered pools for both original and perturbed datasets
                if args.random_pairing_mode == "same_problem":
                    # Members use member_years, perturbed solutions also from member indices
                    non_member_examples = create_non_member_from_perturbed(
                        ds_perturbed, 
                        member_examples=member_examples_list,
                        num_pairs=len(ds_members_subset), 
                        mode=args.random_pairing_mode,
                        seed=args.subset_seed + 1, 
                        verbose=args.verbose,
                        original_dataset=ds_full,
                        member_indices=member_indices
                    )
                elif args.random_pairing_mode == "full_random":
                    # Random problems from non-member pool with their perturbed solutions
                    # Create a modified version that samples from non-member pool
                    nonmember_rng = random.Random(args.subset_seed + 1)
                    selected_indices_in_pool = nonmember_rng.sample(range(len(nonmember_pool_indices)), min(len(ds_members_subset), len(nonmember_pool_indices)))
                    selected_indices = [nonmember_pool_indices[i] for i in selected_indices_in_pool]
                    
                    non_member_examples = []
                    for idx in selected_indices:
                        original_ex = ds_full[idx]
                        perturbed_ex = ds_perturbed[idx]
                        
                        non_member_ex = {
                            "problem": str(original_ex["problem"]).strip(),
                            "solution": str(perturbed_ex["solution"]).strip(),
                            "answer": str(perturbed_ex.get("answer", "")).strip(),
                            "subject": str(perturbed_ex.get("subject", "")).strip(),
                            "level": perturbed_ex.get("level", 0),
                            "unique_id": str(original_ex.get("unique_id", "")).strip(),
                            "is_member": False,
                        }
                        non_member_examples.append(non_member_ex)
                    
                    if args.verbose:
                        print(f"[main] Created {len(non_member_examples)} perturbed non-member examples from year-filtered pool (full_random mode)")
                else:
                    raise ValueError(f"Unknown random_pairing_mode: {args.random_pairing_mode}")
            else:
                non_member_examples = create_non_member_from_perturbed(
                    ds_perturbed, 
                    member_examples=member_examples_list,
                    num_pairs=len(ds_members_subset), 
                    mode=args.random_pairing_mode,
                    seed=args.subset_seed + 1, 
                    verbose=args.verbose,
                    original_dataset=ds_full,
                    member_indices=member_indices
                )
            final_member_indices = member_indices  # No subsampling needed
        elif args.mia_nonmember_method == "separate_dataset":
            print(f"Creating {len(ds_members_subset)} non-member examples from separate dataset...")
            
            # Sample non-members from the separate dataset to match member count
            nonmember_rng = random.Random(args.subset_seed + 1)
            if len(ds_nonmember_full) >= len(ds_members_subset):
                selected_nonmember_indices = nonmember_rng.sample(range(len(ds_nonmember_full)), len(ds_members_subset))
            else:
                # If we have fewer non-members than members, use all of them
                selected_nonmember_indices = list(range(len(ds_nonmember_full)))
            
            selected_nonmember_indices.sort()
            
            # Create non-member examples from selected indices
            non_member_examples = []
            for idx in selected_nonmember_indices:
                ex = ds_nonmember_full[idx]
                non_member_ex = {
                    "problem": str(ex["problem"]).strip(),
                    "solution": str(ex["solution"]).strip(),
                    "answer": str(ex.get("answer", "")).strip(),
                    "subject": str(ex.get("subject", "")).strip(),
                    "level": ex.get("level", 0),
                    "unique_id": str(ex.get("unique_id", "")).strip(),
                    "original_idx": idx,  # Index in the separate dataset
                    "is_member": False,
                    "from_separate_dataset": True  # Flag to identify this source
                }
                non_member_examples.append(non_member_ex)
            
            final_member_indices = member_indices  # No subsampling needed
            
            if args.verbose:
                print(f"[main] Created {len(non_member_examples)} non-member examples from separate dataset")
                print(f"[main] Non-member indices: {selected_nonmember_indices[:10]}..." if len(selected_nonmember_indices) > 10 else f"[main] Non-member indices: {selected_nonmember_indices}")
        else:
            raise ValueError(f"Unknown non-member method: {args.mia_nonmember_method}")
        
        # Prepare augmentation mapping if enabled
        augmentation_map = {}
        if args.augment_target_gt:
            print(f"\n=== Target GT Augmentation ===")
            print(f"Sampling method: {args.augment_sampling_method}")
            print(f"Number of samples per example: {args.augment_num_samples}")
            
            # Collect all solutions from members and non-members
            all_solutions = []
            all_indices = []
            
            # Add member solutions
            for i in range(len(ds_members_subset)):
                all_solutions.append(str(ds_members_subset[i]["solution"]).strip())
                all_indices.append(("member", i))
            
            # Add non-member solutions
            for i, ex in enumerate(non_member_examples):
                all_solutions.append(str(ex["solution"]).strip())
                all_indices.append(("non_member", i))
            
            print(f"Total solution pool size: {len(all_solutions)} (members + non-members)")
            
            # Sample additional solutions for each member
            for i in range(len(ds_members_subset)):
                current_solution = all_solutions[i]
                sampled = sample_additional_solutions(
                    current_solution=current_solution,
                    current_idx=i,
                    all_solutions=all_solutions,
                    all_indices=all_indices,
                    method=args.augment_sampling_method,
                    num_samples=args.augment_num_samples,
                    seed=args.subset_seed
                )
                if sampled:
                    augmentation_map[("member", i)] = sampled
            
            # Sample additional solutions for each non-member
            for i in range(len(non_member_examples)):
                current_solution = all_solutions[len(ds_members_subset) + i]
                pool_idx = len(ds_members_subset) + i
                sampled = sample_additional_solutions(
                    current_solution=current_solution,
                    current_idx=pool_idx,
                    all_solutions=all_solutions,
                    all_indices=all_indices,
                    method=args.augment_sampling_method,
                    num_samples=args.augment_num_samples,
                    seed=args.subset_seed
                )
                if sampled:
                    augmentation_map[("non_member", i)] = sampled
            
            print(f"✅ Augmented {len(augmentation_map)} examples with additional solutions")
            
            # Show example augmentation for first member
            if ("member", 0) in augmentation_map and args.verbose:
                print(f"\nExample augmentation (member 0):")
                print(f"  Original solution: {all_solutions[0][:100]}...")
                for idx, aug_sol in enumerate(augmentation_map[("member", 0)][:2]):
                    print(f"  Additional solution {idx+1}: {aug_sol[:100]}...")
            
            # Clear embedding model cache if it was used
            if args.augment_sampling_method == "embedding":
                _clear_embedding_model_cache()
        
        # Print warning if original ground truth will be removed from non-members
        if args.remove_nonmember_original_gt:
            print(f"\n⚠️  REMOVING ORIGINAL GROUND TRUTH FROM NON-MEMBERS")
            print(f"    Non-members will only have {args.augment_num_samples} augmented solutions (no original solution)")
        
        # Transform member data with augmentation
        def transform_member_with_aug(example, idx):
            aug_sols = augmentation_map.get(("member", idx), None) if augmentation_map else None
            return transform_with_transformed(example, idx, augmented_solutions=aug_sols, exclude_original_solution=False)
        
        # Transform non-member data with augmentation
        def transform_nonmember_with_aug(example, idx):
            aug_sols = augmentation_map.get(("non_member", idx), None) if augmentation_map else None
            # Exclude original solution for non-members if flag is set
            exclude_orig = args.remove_nonmember_original_gt if args.augment_target_gt else False
            return transform_with_transformed(example, idx, augmented_solutions=aug_sols, exclude_original_solution=exclude_orig)
        
        # Apply transformations
        ds_members = ds_members_with_flag.map(transform_member_with_aug, with_indices=True, remove_columns=ds_members_with_flag.column_names)
        if args.verbose:
            print(f"[main] Member data created: {len(ds_members)} records")
        
        ds_non_members_raw = datasets.Dataset.from_list(non_member_examples)
        ds_non_members = ds_non_members_raw.map(transform_nonmember_with_aug, with_indices=True, remove_columns=ds_non_members_raw.column_names)
        if args.verbose:
            print(f"[main] Non-member data created: {len(ds_non_members)} records")
        
        # Load and apply MIA weights if provided
        if args.mia_weights_members and args.mia_weights_nonmembers:
            print(f"\n=== Loading MIA weights ({args.mia_weights_tag}) ===")
            normalized_member_weights, normalized_nonmember_weights = load_and_normalize_mia_weights(
                args.mia_weights_members,
                args.mia_weights_nonmembers,
                verbose=args.verbose
            )
            
            # Verify that the number of weights matches the number of examples
            if len(normalized_member_weights) != len(ds_members):
                raise ValueError(
                    f"Number of member weights ({len(normalized_member_weights)}) does not match "
                    f"number of member examples ({len(ds_members)})"
                )
            if len(normalized_nonmember_weights) != len(ds_non_members):
                raise ValueError(
                    f"Number of non-member weights ({len(normalized_nonmember_weights)}) does not match "
                    f"number of non-member examples ({len(ds_non_members)})"
                )
            
            # Add weights to extra_info for members
            def add_mia_weight_member(example, idx):
                new_example = dict(example)
                if "extra_info" in new_example and isinstance(new_example["extra_info"], dict):
                    new_example["extra_info"] = dict(new_example["extra_info"])
                    new_example["extra_info"]["mia_weight"] = normalized_member_weights[idx]
                    new_example["extra_info"]["mia_weight_tag"] = args.mia_weights_tag
                return new_example
            
            # Add weights to extra_info for non-members
            def add_mia_weight_nonmember(example, idx):
                new_example = dict(example)
                if "extra_info" in new_example and isinstance(new_example["extra_info"], dict):
                    new_example["extra_info"] = dict(new_example["extra_info"])
                    new_example["extra_info"]["mia_weight"] = normalized_nonmember_weights[idx]
                    new_example["extra_info"]["mia_weight_tag"] = args.mia_weights_tag
                return new_example
            
            ds_members = ds_members.map(add_mia_weight_member, with_indices=True)
            ds_non_members = ds_non_members.map(add_mia_weight_nonmember, with_indices=True)
            
            print(f"✅ Added MIA weights ({args.mia_weights_tag}) to {len(ds_members)} members and {len(ds_non_members)} non-members")
        
        # Handle target_gt updates and deduplication if include_target_gt is True
        if args.include_target_gt and args.mia_nonmember_method in ["random_pairing", "perturbed_solution"]:
            print("Checking for matching problems between member and non-member data...")
            
            # Convert datasets to lists for processing
            member_data_list = [ds_members[i] for i in range(len(ds_members))]
            non_member_data_list = [ds_non_members[i] for i in range(len(ds_non_members))]
            
            # Always check for matching problems and update target_gt if matches are found
            updated_member_data, updated_non_member_data = update_target_gt_for_matching_problems(
                member_data_list, non_member_data_list, verbose=args.verbose
            )
            
            # Convert back to datasets
            ds_members = datasets.Dataset.from_list(updated_member_data)
            ds_non_members = datasets.Dataset.from_list(updated_non_member_data)
            
            # Remove duplicate non-member records (will match the problems that had target_gt updated)
            print("Removing duplicate non-member records with same problems as members...")
            ds_non_members = remove_duplicate_problems(ds_members, ds_non_members, verbose=args.verbose)
        
        # Reverse member and non-member labels if requested
        if args.reverse_member:
            print("\n=== REVERSING MEMBER AND NON-MEMBER LABELS ===")
            print("Original members will become non-members, and vice versa")
            
            # Swap is_member flags and ground truth fields in both datasets
            def flip_to_nonmember(example):
                new_example = dict(example)
                if "extra_info" in new_example and isinstance(new_example["extra_info"], dict):
                    new_example["extra_info"] = dict(new_example["extra_info"])
                    new_example["extra_info"]["is_member"] = False
                    
                    # Swap member_ground_truth and nonmember_ground_truth if they exist
                    if "member_ground_truth" in new_example["extra_info"] and "nonmember_ground_truth" in new_example["extra_info"]:
                        member_gt = new_example["extra_info"]["member_ground_truth"]
                        nonmember_gt = new_example["extra_info"]["nonmember_ground_truth"]
                        new_example["extra_info"]["member_ground_truth"] = nonmember_gt
                        new_example["extra_info"]["nonmember_ground_truth"] = member_gt
                return new_example
            
            def flip_to_member(example):
                new_example = dict(example)
                if "extra_info" in new_example and isinstance(new_example["extra_info"], dict):
                    new_example["extra_info"] = dict(new_example["extra_info"])
                    new_example["extra_info"]["is_member"] = True
                    
                    # Swap member_ground_truth and nonmember_ground_truth if they exist
                    if "member_ground_truth" in new_example["extra_info"] and "nonmember_ground_truth" in new_example["extra_info"]:
                        member_gt = new_example["extra_info"]["member_ground_truth"]
                        nonmember_gt = new_example["extra_info"]["nonmember_ground_truth"]
                        new_example["extra_info"]["member_ground_truth"] = nonmember_gt
                        new_example["extra_info"]["nonmember_ground_truth"] = member_gt
                return new_example
            
            # Apply the flips
            ds_members_flipped = ds_members.map(flip_to_nonmember)
            ds_non_members_flipped = ds_non_members.map(flip_to_member)
            
            # Swap the datasets for parquet generation
            ds_members = ds_non_members_flipped  # For the combined dataset
            ds_non_members = ds_members_flipped  # For the combined dataset
            
            # For JSONL generation, swap the raw data references
            ds_members_subset_for_jsonl = non_member_examples  # Original non-members written as members
            non_member_examples_for_jsonl = [dict(ex) for ex in ds_members_subset]  # Original members written as non-members
            
            if args.verbose:
                print(f"[main] After reversal: {len(ds_members)} members (originally non-members), {len(ds_non_members)} non-members (originally members)")
        else:
            # No reversal - keep datasets as-is for JSONL generation
            ds_members_subset_for_jsonl = ds_members_subset
            non_member_examples_for_jsonl = non_member_examples
        
        # Combine member and non-member data for the final training set
        ds_combined = datasets.concatenate_datasets([ds_members, ds_non_members])
        
        # Shuffle the combined dataset
        ds_transformed = ds_combined.shuffle(seed=args.subset_seed)
        
        # Verify MIA weights are preserved after all processing
        if args.mia_weights_members and args.mia_weights_nonmembers:
            weights_found = 0
            for i in range(min(5, len(ds_transformed))):  # Check first 5 examples
                if "extra_info" in ds_transformed[i] and isinstance(ds_transformed[i]["extra_info"], dict):
                    if "mia_weight" in ds_transformed[i]["extra_info"]:
                        weights_found += 1
            
            if weights_found > 0:
                print(f"✅ Verified: MIA weights preserved in final dataset (checked {weights_found}/5 examples)")
            else:
                print(f"⚠️  WARNING: MIA weights NOT found in final dataset! Check processing pipeline.")
        
        if args.verbose:
            if args.include_target_gt and args.mia_nonmember_method in ["random_pairing", "perturbed_solution"]:
                print(f"[main] Created combined dataset: {len(ds_members)} members + {len(ds_non_members)} non-members (after deduplication) = {len(ds_transformed)} total")
            else:
                print(f"[main] Created combined dataset: {len(ds_members)} members + {len(ds_non_members)} non-members = {len(ds_transformed)} total")
    else:
        # Standard mode: Just transform the subsampled dataset
        ds_transformed = ds_members_subset.map(transform_fn, with_indices=True, remove_columns=ds_members_subset.column_names)

    if args.verbose:
        print(f"\n[main] Transformation completed - {len(ds_transformed)} records created")
        print(f"[main] Data source: {ds_transformed[0]['data_source'] if len(ds_transformed) > 0 else 'N/A'}")
        print(f"[main] Match type: {args.match_type}")
        if args.mia:
            member_count = sum(1 for record in ds_transformed if record.get('extra_info', {}).get('is_member', False))
            non_member_count = len(ds_transformed) - member_count
            print(f"[main] MIA data: {member_count} members, {non_member_count} non-members")

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
        indices_path = os.path.join(output_dir, f"{args.output_name}_indices.json")

        print(f"\n=== Generating MIA JSONL files ===")
        
        # Build member rows (using potentially reversed data)
        members_rows = []
        if isinstance(ds_members_subset_for_jsonl, list):
            # When reversed, this will be non_member_examples (a list)
            member_data_to_iterate = ds_members_subset_for_jsonl
        else:
            # When not reversed, this will be ds_members_subset (a dataset)
            member_data_to_iterate = [ds_members_subset_for_jsonl[i] for i in range(len(ds_members_subset_for_jsonl))]
        
        for i, ex in enumerate(member_data_to_iterate):
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

        # Build non-member rows (using potentially reversed data)
        nonmembers_rows = []
        
        for i, ex in enumerate(non_member_examples_for_jsonl):
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
            elif args.mia_nonmember_method == "perturbed_solution":
                row["pair_type"] = "perturbed"
            elif args.mia_nonmember_method == "separate_dataset":
                row["pair_type"] = "separate_dataset"
                row["original_idx"] = ex.get("original_idx", -1)
                row["nonmember_dataset_path"] = args.nonmember_dataset_path
            
            nonmembers_rows.append(row)

        _write_jsonl(members_path, members_rows)
        _write_jsonl(nonmembers_path, nonmembers_rows)
        
        # Save member and non-member indices for verification
        index_info = {
            "member_indices": final_member_indices,  # Original indices from full dataset
            "member_seed": args.subset_seed,
            "member_size": len(final_member_indices),
            "nonmember_method": args.mia_nonmember_method,
            "reverse_member": args.reverse_member,  # Track if labels were reversed
            "dataset_info": {
                "dataset_path": args.dataset_path,
                "dataset_split": args.dataset_split,
                "total_dataset_size": len(ds_full)
            }
        }
        
        # Add year filtering information if enabled
        if args.filter_by_year:
            index_info["year_filtering"] = {
                "enabled": True,
                "member_years": args.member_years,
                "nonmember_years": args.nonmember_years,
                "member_pool_size": len(member_pool_indices),
                "nonmember_pool_size": len(nonmember_pool_indices),
                "member_pool_indices": member_pool_indices,
                "nonmember_pool_indices": nonmember_pool_indices
            }
        
        # Add non-member specific info
        if args.mia_nonmember_method == "unused_examples":
            # Get non-member indices (unused examples)
            nonmember_indices = [ex.get("original_idx") for ex in non_member_examples if "original_idx" in ex]
            nonmember_indices = [idx for idx in nonmember_indices if idx is not None]
            nonmember_indices.sort()
            index_info["nonmember_indices"] = nonmember_indices
            index_info["nonmember_size"] = len(nonmember_indices)
        elif args.mia_nonmember_method == "separate_dataset":
            # Get non-member indices from separate dataset
            nonmember_indices = [ex.get("original_idx") for ex in non_member_examples if "original_idx" in ex]
            nonmember_indices = [idx for idx in nonmember_indices if idx is not None]
            nonmember_indices.sort()
            index_info["nonmember_indices"] = nonmember_indices
            index_info["nonmember_size"] = len(nonmember_indices)
            index_info["nonmember_dataset_path"] = args.nonmember_dataset_path
            index_info["nonmember_dataset_split"] = args.nonmember_dataset_split
            if args.nonmember_transformed_dataset_path:
                index_info["nonmember_transformed_dataset_path"] = args.nonmember_transformed_dataset_path
                index_info["nonmember_transformed_dataset_split"] = args.nonmember_transformed_dataset_split
        elif args.mia_nonmember_method in ["random_pairing", "perturbed_solution"]:
            # For these methods, store the problem and solution indices
            problem_indices = [ex.get("original_problem_idx") for ex in non_member_examples if "original_problem_idx" in ex]
            solution_indices = [ex.get("original_solution_idx") for ex in non_member_examples if "original_solution_idx" in ex]
            index_info["nonmember_problem_indices"] = problem_indices
            index_info["nonmember_solution_indices"] = solution_indices
            index_info["nonmember_size"] = len(non_member_examples)
        
        with open(indices_path, "w") as f:
            json.dump(index_info, f, indent=2)
        
        print(f"Wrote MIA JSONL files:")
        print(f"  Members ({len(members_rows)} examples): {members_path}")
        print(f"  Non-members ({len(nonmembers_rows)} examples): {nonmembers_path}")
        print(f"  Indices ({args.mia_nonmember_method} method): {indices_path}")
        if args.reverse_member:
            print(f"  NOTE: Labels were REVERSED - members.jsonl contains original non-members, nonmembers.jsonl contains original members")
        
        # Print index info
        print(f"\n=== MIA Index Information ===")
        print(f"Member indices: {final_member_indices[:10]}{'...' if len(final_member_indices) > 10 else ''}")
        if args.mia_nonmember_method == "unused_examples" and "nonmember_indices" in index_info:
            print(f"Non-member indices: {index_info['nonmember_indices'][:10]}{'...' if len(index_info['nonmember_indices']) > 10 else ''}")
        
        # Verification
        if args.verbose:
            print(f"\n=== MIA Data Verification ===")
            print(f"Members: {len(members_rows)} examples (original problem-solution pairs)")
            print(f"Non-members: {len(nonmembers_rows)} examples ({args.mia_nonmember_method} method)")
            if len(members_rows) > 0:
                sample_member = members_rows[0]
                print(f"Sample member problem: {sample_member['problem'][:80]}...")
            if len(nonmembers_rows) > 0:
                sample_nonmember = nonmembers_rows[0]
                print(f"Sample non-member problem: {sample_nonmember['problem'][:80]}...")
                print(f"Non-member method: {args.mia_nonmember_method}")
            
            # Check for overlap (should be none for unused_examples method)
            if args.mia_nonmember_method == "unused_examples" and "nonmember_indices" in index_info:
                member_set = set(final_member_indices)
                nonmember_set = set(index_info["nonmember_indices"])
                overlap = member_set.intersection(nonmember_set)
                if overlap:
                    print(f"❌ ERROR: Found {len(overlap)} overlapping indices: {list(overlap)[:5]}...")
                else:
                    print(f"✅ VERIFIED: No overlap between member and non-member indices")


if __name__ == "__main__":
    main()
