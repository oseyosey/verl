from __future__ import annotations

"""Lexical similarity–based reward functions.

This module offers a comprehensive way to evaluate lexical similarity between
model responses (``solution_str``) and ground-truth answers (``ground_truth``).
The main entry-point is :pyfunc:`compute_score`, which follows the interface
expected by verl's reward loading utilities.

Key features
------------
* **Six lexical metrics** available:
  - ``lexical_token_overlap``: Jaccard similarity (0-1)
  - ``lexical_lcs_ratio``: Normalized LCS ratio by reference length (0-1)
  - ``lexical_lcs_ratio_cand``: Normalized LCS ratio by candidate length (0-1)
  - ``length_ratio``: Token length ratio (candidate/reference)
  - ``lexical_ngram_coverage``: N-gram coverage normalized by candidate (0-1)
  - ``lexical_ngram_coverage_ref``: N-gram coverage normalized by reference (0-1)
* **Flexible metric selection** via ``metric_profile`` parameter
* **Weighted aggregation** of multiple metrics with configurable weights
* **Length penalty** support to discourage outputs that are too short or too long
* **MIA weighting** support to apply membership inference weights during RL training
* **Concurrent processing** support for efficient batch evaluation
* **Qwen2.5-Math tokenization** for long sequence support (32k+ tokens) and math text optimization
* **Extensible** design allows custom metric profiles

MIA Weighting
~~~~~~~~~~~~~
The module supports MIA (Membership Inference Attack) weighting for RL training:
- Configure via ``use_mia_weighting`` and ``mia_invert_weights`` in metric profiles
- Handles both simple mode (``mia_weight``) and complex mode (``member_mia_weight``, ``nonmember_mia_weight``)
- Weights are extracted from ``extra_info`` and applied to final scores

Example
~~~~~~~
>>> from verl.utils.reward_score.lexical import compute_score
>>> # Default profile using 3 metrics with equal weights
>>> compute_score(
...     data_source="dummy",
...     solution_str="Cats are great pets.",
...     ground_truth="Cats make wonderful companions."
... )
0.333...

>>> # Using a specific metric
>>> compute_score(
...     data_source="dummy",
...     solution_str="the quick brown fox",
...     ground_truth="quick brown fox jumps",
...     metric_profile="lexical_token_overlap"
... )
0.428...

>>> # Using MIA-weighted profile
>>> compute_score(
...     data_source="dummy",
...     solution_str="Cats are great pets.",
...     ground_truth="Cats make wonderful companions.",
...     extra_info={"mia_weight": 0.8},
...     metric_profile="duo_v2_ratio_penalty_1.25_mia"
... )
0.666...
"""

import pdb  # * Hacky way to debug the verl codebase (ray cluster)

from typing import Callable, List, Dict, Optional, Any, Tuple
import re
import warnings
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from difflib import SequenceMatcher

# Import tqdm for progress tracking
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# Import tokenizer - using Qwen2.5-Math for long sequence support (32k+ tokens)
try:
    from transformers import AutoTokenizer
    _DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
    _HAS_TOKENIZER = True
except ImportError:
    _DEFAULT_TOKENIZER = None
    _HAS_TOKENIZER = False
    warnings.warn(
        "transformers not available. Lexical metrics will use regex fallback.",
        RuntimeWarning
    )

# Import n-gram coverage
try:
    from ...ddrl.utils_rl.ngram_coverage import compute_ngram_coverage
    _HAS_NGRAM_COVERAGE = True
except ImportError:
    # Try alternative import paths
    try:
        from ddrl.utils_rl.ngram_coverage import compute_ngram_coverage
        _HAS_NGRAM_COVERAGE = True
    except ImportError:
        _HAS_NGRAM_COVERAGE = False
        warnings.warn(
            "ngram_coverage not available. N-gram coverage metric will be skipped.",
            RuntimeWarning
        )

# Legacy BM25 support (kept for backward compatibility)
try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_BM25 = True
except ModuleNotFoundError:  # pragma: no cover – runtime fallback
    _HAS_BM25 = False

__all__: List[str] = [
    "compute_score",
    "compute_score_batched",
]

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_NUM_WORKERS = 32  # Number of parallel workers for batch processing

# Default length penalty configuration (same as embedding_remote.py)
DEFAULT_LENGTH_PENALTY_TYPE = "none"
DEFAULT_LENGTH_THRESHOLD = 1.5

# -----------------------------------------------------------------------------
# Metric profiles configuration
# -----------------------------------------------------------------------------

# Default metric profiles with weights
METRIC_PROFILES = {
    # Default profile: average of 3 key metrics with equal weights
    "duo_v1_ratio_penalty_1.25": {
        "metrics": ["lexical_lcs_ratio_cand", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.25
    },
    "duo_v2_ratio_penalty_1.25": {
        "metrics": ["lexical_lcs_ratio", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.25
    },
    "duo_v2_quadratic_penalty_1.25": {
        "metrics": ["lexical_lcs_ratio", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0],
        "length_penalty_type": "quadratic",
        "length_threshold": 1.25
    },
    "trio_v1": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio_cand", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0]
    },
    # Option A: Reference-normalized metrics (recommended to avoid short-sentence bias)
    "trio_v2": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0]
    },
    # Option B: Original metrics with length penalty
    "trio_v1_ratio_penalty_1.25": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio_cand", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",  # Options: none, ratio, sqrt, log, quadratic, exponential
        "length_threshold": 1.25  # Threshold for length penalty (default: 1.25)
    },
    "trio_v1_quadratic_penalty_1.25": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio_cand", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "quadratic",  # Options: none, ratio, sqrt, log, quadratic, exponential
        "length_threshold": 1.25  # Threshold for length penalty (default: 1.25)
    },
    # Trio v2 with length penalty for extra safety
    "trio_v2_ratio_penalty_1.25": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.25
    },
    "trio_v3_ratio_penalty_1.25": {
        "metrics": ["length_ratio", "lexical_lcs_ratio", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.25
    },
    # Comprehensive profile using all metrics
    "comprehensive": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio", "lexical_lcs_ratio_cand", 
                   "length_ratio", "lexical_ngram_coverage", "lexical_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0, 0.5, 1.0, 1.0]  # Lower weight for length_ratio
    },
    # Individual metrics (for backward compatibility and specific use cases)
    "lexical_token_overlap": {
        "metrics": ["lexical_token_overlap"],
        "weights": [1.0]
    },
    "lexical_lcs_ratio": {
        "metrics": ["lexical_lcs_ratio"],
        "weights": [1.0]
    },
    "lexical_lcs_ratio_cand": {
        "metrics": ["lexical_lcs_ratio_cand"],
        "weights": [1.0]
    },
    "length_ratio": {
        "metrics": ["length_ratio"],
        "weights": [1.0]
    },
    "lexical_ngram_coverage": {
        "metrics": ["lexical_ngram_coverage"],
        "weights": [1.0]
    },
    "lexical_ngram_coverage_ref": {
        "metrics": ["lexical_ngram_coverage_ref"],
        "weights": [1.0]
    },
    # Legacy compatibility mappings
    "token_ratio": {
        "metrics": ["lexical_token_overlap"],
        "weights": [1.0]
    },
    "ordered_token": {
        "metrics": ["lexical_lcs_ratio"],
        "weights": [1.0]
    },
    # MIA-weighted profiles (examples)
    "trio_v1_ratio_penalty_1.25_mia": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio_cand", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.25,
        "use_mia_weighting": True,
        "mia_invert_weights": True,  # Lower MIA score = more likely member = higher weight
    },
}

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _tokenize(text: str, max_tokens: Optional[int] = None) -> List[str]:
    """Tokenise text into a list of tokens using Qwen2.5-Math tokenizer.
    
    Uses Qwen2.5-Math-7B-Instruct tokenizer which supports long sequences (32k+ tokens)
    and is optimized for mathematical text. Falls back to regex tokenization if the
    tokenizer is not available.
    
    Args:
        text: Text to tokenize
        max_tokens: Maximum tokens to return (for truncation)
        
    Returns:
        List of tokens
    """
    if _HAS_TOKENIZER and _DEFAULT_TOKENIZER is not None:
        # Use Qwen2.5-Math tokenizer for long sequence support
        return _DEFAULT_TOKENIZER.tokenize(
            text,
            max_length=max_tokens,
            truncation=True,
        )
    else:
        # Fallback to regex tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        if max_tokens is not None and len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokens


def _compute_lexical_metrics(reference: str, candidate: str, metrics_to_compute: Optional[set] = None) -> Dict[str, float]:
    """Compute lexical metrics between reference and candidate.
    
    This function computes up to 6 different lexical metrics, matching the
    implementation in llm_judge_remote.py for consistency across the codebase.
    
    Args:
        reference: Ground truth text
        candidate: Candidate text to evaluate
        metrics_to_compute: Set of metric names to compute. If None, computes all metrics.
                           Valid names: 'lexical_token_overlap', 'lexical_lcs_ratio', 
                           'lexical_lcs_ratio_cand', 'length_ratio', 'lexical_ngram_coverage',
                           'lexical_ngram_coverage_ref'
        
    Returns:
        Dict with requested metrics:
        - lexical_token_overlap: Jaccard similarity (0-1)
        - lexical_lcs_ratio: Normalized LCS ratio by reference length (0-1)
        - lexical_lcs_ratio_cand: Normalized LCS ratio by candidate length (0-1)
        - length_ratio: Token length ratio (candidate/reference)
        - lexical_ngram_coverage: N-gram coverage normalized by candidate (0-1)
        - lexical_ngram_coverage_ref: N-gram coverage normalized by reference (0-1)
    """
    # If no specific metrics requested, compute all
    if metrics_to_compute is None:
        metrics_to_compute = {
            'lexical_token_overlap', 'lexical_lcs_ratio', 'lexical_lcs_ratio_cand',
            'length_ratio', 'lexical_ngram_coverage', 'lexical_ngram_coverage_ref'
        }
    
    result = {}
    
    # Determine if we need tokenization (needed for most metrics)
    needs_tokenization = any(m in metrics_to_compute for m in [
        'lexical_token_overlap', 'lexical_lcs_ratio', 'lexical_lcs_ratio_cand', 'length_ratio'
    ])
    
    if needs_tokenization:
        # Tokenize both texts
        ref_tokens = _tokenize(reference)
        cand_tokens = _tokenize(candidate)
    
    # 1. Lexical token overlap (Jaccard similarity)
    if 'lexical_token_overlap' in metrics_to_compute:
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        intersection = ref_set & cand_set
        union = ref_set | cand_set
        result['lexical_token_overlap'] = len(intersection) / len(union) if union else 0.0
    
    # 2. Lexical LCS ratio (both normalizations)
    if 'lexical_lcs_ratio' in metrics_to_compute or 'lexical_lcs_ratio_cand' in metrics_to_compute:
        if not ref_tokens or not cand_tokens:
            if 'lexical_lcs_ratio' in metrics_to_compute:
                result['lexical_lcs_ratio'] = 0.0
            if 'lexical_lcs_ratio_cand' in metrics_to_compute:
                result['lexical_lcs_ratio_cand'] = 0.0
        else:
            # Compute LCS length using dynamic programming
            m, n = len(ref_tokens), len(cand_tokens)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if ref_tokens[i-1] == cand_tokens[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            lcs_length = dp[m][n]
            # Normalize by ground truth length (reference)
            if 'lexical_lcs_ratio' in metrics_to_compute:
                result['lexical_lcs_ratio'] = lcs_length / len(ref_tokens)
            # Normalize by candidate length
            if 'lexical_lcs_ratio_cand' in metrics_to_compute:
                result['lexical_lcs_ratio_cand'] = lcs_length / len(cand_tokens)
    
    # 3. Length ratio (candidate / reference)
    if 'length_ratio' in metrics_to_compute:
        if not ref_tokens:
            result['length_ratio'] = 0.0 if cand_tokens else 1.0
        else:
            result['length_ratio'] = len(cand_tokens) / len(ref_tokens)
    
    # 4. N-gram coverage (normalized by candidate)
    if 'lexical_ngram_coverage' in metrics_to_compute:
        if _HAS_NGRAM_COVERAGE:
            result['lexical_ngram_coverage'] = compute_ngram_coverage(candidate, reference, min_ngram=3, normalize_by="candidate")
        else:
            result['lexical_ngram_coverage'] = 0.0
    
    # 5. N-gram coverage (normalized by reference)
    if 'lexical_ngram_coverage_ref' in metrics_to_compute:
        if _HAS_NGRAM_COVERAGE:
            result['lexical_ngram_coverage_ref'] = compute_ngram_coverage(candidate, reference, min_ngram=3, normalize_by="reference")
        else:
            result['lexical_ngram_coverage_ref'] = 0.0
    
    return result


def _compute_length_penalty(
    reference: str, 
    candidate: str,
    penalty_type: str = "none",
    threshold: float = 1.5
) -> float:
    """
    Compute length penalty factor (same as embedding_remote.py).
    Returns a value between 0 and 1, where 1 means no penalty.
    
    Args:
        reference: Ground truth text
        candidate: Candidate text to evaluate
        penalty_type: Type of penalty to apply. Options:
            - "none": No penalty (default)
            - "ratio": Linear penalty based on length ratio
            - "sqrt": Square root penalty (softer)
            - "log": Logarithmic penalty (softest)
            - "quadratic": Quadratic penalty (harsher)
            - "exponential": Exponential penalty (harshest)
        threshold: Length ratio threshold (default: 1.5)
            If candidate length is outside [ref_len/threshold, ref_len*threshold], penalty applies
    
    Returns:
        Penalty factor between 0 and 1
    """
    ref_len = len(reference.split())
    cand_len = len(candidate.split())
    
    # Calculate both ratios to handle both longer and shorter outputs
    if ref_len == 0:
        return 1.0 if cand_len == 0 else 0.0
    
    ratio_short = cand_len / ref_len  # < 1/threshold means too short
    ratio_long = ref_len / cand_len if cand_len > 0 else 0.0   # < 1/threshold means too long
    
    # No penalty if output length is within acceptable range
    if ratio_short >= 1/threshold and ratio_short <= threshold:
        return 1.0
    
    # Use the more extreme ratio for penalty calculation
    ratio = min(ratio_long, ratio_short)
    
    if penalty_type == "none":
        return 1.0
    elif penalty_type == "ratio":
        return min(1.0, ratio)
    elif penalty_type == "sqrt":
        return min(1.0, ratio ** 0.5)
    elif penalty_type == "log":
        return min(1.0, math.log(1 + min(ref_len, cand_len)) / math.log(1 + max(ref_len, cand_len)))
    elif penalty_type == "quadratic":
        return min(1.0, ratio ** 2)
    elif penalty_type == "exponential":
        return min(1.0, math.exp(-(1 - ratio)))
    else:
        warnings.warn(f"Unknown length penalty type: {penalty_type}, using 'none'")
        return 1.0


def _aggregate_metrics(
    metrics: Dict[str, float], 
    profile: Dict[str, Any], 
    reference: Optional[str] = None,
    candidate: Optional[str] = None
) -> float:
    """Aggregate multiple metrics using weighted average.
    
    Args:
        metrics: Dictionary of computed metric values
        profile: Metric profile containing 'metrics' list and 'weights' list
        reference: Ground truth text (optional, needed for length penalty)
        candidate: Candidate text (optional, needed for length penalty)
        
    Returns:
        Weighted average score in [0, 1]
    """
    selected_metrics = profile["metrics"]
    weights = profile["weights"]
    
    if len(selected_metrics) != len(weights):
        raise ValueError(f"Metrics and weights must have same length: {len(selected_metrics)} != {len(weights)}")
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for metric_name, weight in zip(selected_metrics, weights):
        if metric_name in metrics:
            value = metrics[metric_name]
            # Special handling for length_ratio: convert to similarity score
            if metric_name == "length_ratio":
                # Map length ratio to similarity: 1.0 when ratio=1, decreases as ratio diverges
                value = 1.0 / (1.0 + abs(value - 1.0))
            weighted_sum += value * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    # Compute base score
    base_score = weighted_sum / total_weight
    
    # Apply length penalty if configured and we have the necessary texts
    length_penalty_type = profile.get("length_penalty_type", "none")
    if length_penalty_type != "none" and reference is not None and candidate is not None:
        length_threshold = profile.get("length_threshold", DEFAULT_LENGTH_THRESHOLD)
        penalty = _compute_length_penalty(reference, candidate, length_penalty_type, length_threshold)
        return base_score * penalty
    
    return base_score


def _compute_score_single(reference: str, candidate: str, profile: Dict[str, Any]) -> float:
    """Compute score for a single reference-candidate pair.
    
    Helper function for parallel processing.
    
    Args:
        reference: Ground truth text
        candidate: Candidate text
        profile: Metric profile configuration
        
    Returns:
        Similarity score in [0, 1]
    """
    metrics = _compute_lexical_metrics(reference, candidate, set(profile["metrics"]))
    return _aggregate_metrics(metrics, profile, reference, candidate)


def _compute_scores_parallel(
    references: List[str],
    candidates: List[str],
    profile: Dict[str, Any],
    num_workers: int = DEFAULT_NUM_WORKERS,
    show_progress: bool = False,
    desc: str = "Computing lexical scores"
) -> List[float]:
    """Compute lexical scores for multiple pairs in parallel.
    
    This function parallelizes the lexical metrics computation across multiple workers,
    which is especially beneficial for large batches and expensive metrics like n-gram coverage.
    
    Args:
        references: List of ground truth texts
        candidates: List of candidate texts to evaluate
        profile: Metric profile configuration
        num_workers: Number of parallel workers (default: 32)
        show_progress: Whether to show progress bar with throughput metrics
        desc: Description for the progress bar
        
    Returns:
        List of similarity scores
    """
    # For small batches or when num_workers=1, use sequential processing
    if len(references) < num_workers * 2 or num_workers <= 1:
        # Logging to help diagnose why we might be running sequentially
        try:
            import logging
            logging.getLogger(__name__).info(
                "lexical._compute_scores_parallel: using SEQUENTIAL path (pairs=%d, num_workers=%d, threshold=%d)",
                len(references), num_workers, num_workers * 2,
            )
        except Exception:
            pass
        iterator = zip(references, candidates)
        if show_progress and _HAS_TQDM:
            iterator = tqdm(list(iterator), desc=desc, unit="sample", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        return [
            _compute_score_single(ref, cand, profile)
            for ref, cand in iterator
        ]
    
    # Parallel processing for larger batches
    try:
        import logging
        logging.getLogger(__name__).info(
            "lexical._compute_scores_parallel: using PARALLEL ProcessPool path (pairs=%d, num_workers=%d)",
            len(references), num_workers,
        )
    except Exception:
        pass
    scores = [0.0] * len(references)
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        for idx, (ref, cand) in enumerate(zip(references, candidates)):
            future = executor.submit(_compute_score_single, ref, cand, profile)
            future_to_index[future] = idx
        
        # Collect results with progress tracking
        pbar = None
        if show_progress and _HAS_TQDM:
            pbar = tqdm(total=len(references), desc=desc, unit="sample",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        completed = 0
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                scores[idx] = future.result()
            except Exception as e:
                # Log error but continue with 0.0 score
                import logging
                logging.error(f"Error computing score at index {idx}: {e}")
                scores[idx] = 0.0
            
            completed += 1
            if pbar is not None:
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
    
    # Print timing summary if progress was shown
    if show_progress:
        elapsed = time.time() - start_time
        throughput = len(references) / elapsed if elapsed > 0 else 0
        print(f"✓ Computed {len(references)} lexical scores in {elapsed:.2f}s ({throughput:.1f} samples/sec, {num_workers} workers)")
    
    return scores

# -----------------------------------------------------------------------------
# Utility: Extract MIA weight from extra_info
# -----------------------------------------------------------------------------


def _extract_mia_weight(ref: str, extra_info: dict | None) -> Optional[float]:
    """Extract MIA weight for a specific reference from extra_info.
    
    Handles two modes:
    1. Simple mode (unused_examples): Single mia_weight field applies to all references
    2. Complex mode (perturbed_solution): Separate member_mia_weight and nonmember_mia_weight
       based on which ground truth (member_ground_truth or nonmember_ground_truth) matches
    
    Args:
        ref: The reference string being evaluated
        extra_info: Extra information dict containing MIA weight fields
        
    Returns:
        MIA weight (float in [0, 1]) or None if not available
    """
    if not extra_info or not isinstance(extra_info, dict):
        return None
    
    # Simple mode: Single mia_weight field (unused_examples mode)
    if "mia_weight" in extra_info:
        return float(extra_info["mia_weight"])
    
    # Complex mode: Separate weights for member and non-member ground truths (perturbed_solution mode)
    # Check if we have the necessary fields
    has_member_fields = "member_ground_truth" in extra_info and "member_mia_weight" in extra_info
    has_nonmember_fields = "nonmember_ground_truth" in extra_info and "nonmember_mia_weight" in extra_info
    
    if has_member_fields or has_nonmember_fields:
        # Match reference to ground truth and return corresponding weight
        if has_member_fields:
            member_gt = extra_info["member_ground_truth"]
            if ref == member_gt:
                return float(extra_info["member_mia_weight"])
        
        if has_nonmember_fields:
            nonmember_gt = extra_info["nonmember_ground_truth"]
            if ref == nonmember_gt:
                return float(extra_info["nonmember_mia_weight"])
    
    return None


# -----------------------------------------------------------------------------
# Utility: filter reference list based on *extra_info*
# -----------------------------------------------------------------------------


def _filter_refs(refs: List[str], extra_info: dict | None) -> List[str]:
    """Filter references (same as llm_judge_remote.py).
    
    BUG FIX: Previously used [r for r in refs if r in tgt] which only returned
    refs that existed in BOTH the original refs AND target_gt. This caused issues
    when target_gt contained references not in the original refs list.
    
    NEW BEHAVIOR: target_gt is now authoritative - we return exactly what's in
    target_gt, not filtered by the original refs. This ensures all specified
    target ground truths are used for lexical scoring.
    """
    if not extra_info or not isinstance(extra_info, dict):
        return refs

    # 1. Exact target string(s) - target_gt is authoritative (returns target_gt directly)
    tgt = extra_info.get("target_gt")
    if isinstance(tgt, str):
        # Return the target string directly (not filtered by refs)
        return [tgt]
    elif isinstance(tgt, list):
        # Return the target list directly (not filtered by refs)
        return tgt

    # 2. Last prompt token heuristic
    if extra_info.get("filter_gt_by_prompt_token") and "prompt" in extra_info:
        prompt_txt = str(extra_info["prompt"]).strip()
        if prompt_txt:
            last_tok = prompt_txt.split()[-1].lower()
            subset = [r for r in refs if last_tok in _tokenize(r)]
            if subset:
                return subset

    return refs


# -----------------------------------------------------------------------------
# Public API expected by verl
# -----------------------------------------------------------------------------

def compute_score(
    data_source: str | List[str] | None = None,
    solution_str: str | List[str] | None = None,
    ground_truth: str | List[str] | None = None,
    extra_info: dict | List[dict] | None = None,
    *,
    # Batch-style parameters used by BatchRewardManager.
    data_sources: List[str] | None = None,
    solution_strs: List[str] | None = None,
    ground_truths: List[str | List[str]] | None = None,
    extra_infos: List[dict | None] | None = None,
    metric_profile: str = "default",
) -> float | List[float]:
    """Return lexical similarity score between *solution_str* and *ground_truth*.

    Parameters
    ----------
    data_source
        Ignored – kept to satisfy the expected signature.
    solution_str
        The model-generated answer.
    ground_truth
        Reference answer(s). A single string or a list of strings. When a list
        is provided, the maximum score across references is returned.
    extra_info
        Extra information supporting:
        - target_gt: String or list of strings for exact ground truth matching
        - filter_gt_by_prompt_token: Boolean to filter by last prompt token
        - prompt: Prompt text for token filtering
        - metric_profile: Override the metric_profile parameter
        - custom_weights: List of weights for the selected metrics
        - length_penalty_type: Override length penalty type ("none", "ratio", "sqrt", "log", "quadratic", "exponential")
        - length_threshold: Override length threshold (float, default: 1.5)
        - num_workers: Number of parallel workers for batch processing (default: 32)
        - show_progress: Boolean to show progress bar with throughput metrics (default: False)
        - mia_weight: MIA weight for simple mode (unused_examples) in [0, 1]
        - member_mia_weight: MIA weight for member ground truth (perturbed_solution mode)
        - nonmember_mia_weight: MIA weight for non-member ground truth (perturbed_solution mode)
        - member_ground_truth: Member ground truth string (perturbed_solution mode)
        - nonmember_ground_truth: Non-member ground truth string (perturbed_solution mode)
    metric_profile
        The metric profile to use. Available options:
        - **default**: Average of token_overlap, lcs_ratio_cand, ngram_coverage (may favor short outputs)
        - **trio_v2**: Average of token_overlap, lcs_ratio, ngram_coverage_ref (reference-normalized, recommended)
        - **default_with_length_penalty**: Default metrics with ratio-based length penalty
        - **trio_v2_with_length_penalty**: Trio v2 metrics with ratio-based length penalty
        - **lexical_token_overlap**: Jaccard similarity only
        - **lexical_lcs_ratio**: LCS normalized by reference only
        - **lexical_lcs_ratio_cand**: LCS normalized by candidate only
        - **length_ratio**: Length ratio only
        - **lexical_ngram_coverage**: N-gram coverage by candidate only
        - **lexical_ngram_coverage_ref**: N-gram coverage by reference only
        - **comprehensive**: All metrics with weighted average
        - Legacy names supported: **token_ratio**, **ordered_token**
        
        MIA weighting can be configured in metric profiles:
        - **use_mia_weighting**: Boolean to enable MIA weighting (default: False)
        - **mia_invert_weights**: Boolean to invert weights (1 - weight) (default: False)

    Returns
    -------
    float
        A value in the range [0, 1] – higher means more similar.
        When MIA weighting is enabled, the score is multiplied by the MIA weight.
    """

    # Extract configuration from extra_info
    config = {}
    if isinstance(extra_info, dict):
        config = extra_info.copy()
    elif extra_infos is not None and len(extra_infos) > 0:
        # In batched mode, get config from first extra_info
        first_extra_info = extra_infos[0]
        if isinstance(first_extra_info, dict):
            config = first_extra_info.copy()
    
    # Override metric_profile if specified in config
    if "metric_profile" in config:
        metric_profile = config["metric_profile"]
    
    # Get metric profile configuration
    if metric_profile not in METRIC_PROFILES:
        # Check if it's a legacy metric name that needs special handling
        if metric_profile == "bm25":
            warnings.warn("BM25 metric is deprecated. Using lexical_token_overlap instead.", DeprecationWarning)
            metric_profile = "lexical_token_overlap"
        elif metric_profile == "ratio":
            warnings.warn("Character ratio metric is deprecated. Using default profile instead.", DeprecationWarning)
            metric_profile = "default"
        else:
            raise ValueError(
                f"Unknown metric profile '{metric_profile}'. Available: {list(METRIC_PROFILES.keys())}"
            )
    
    profile = METRIC_PROFILES[metric_profile].copy()
    
    # Allow custom weights if provided
    if "custom_weights" in config and isinstance(config["custom_weights"], list):
        if len(config["custom_weights"]) == len(profile["metrics"]):
            profile["weights"] = config["custom_weights"]
        else:
            warnings.warn(
                f"Custom weights length {len(config['custom_weights'])} doesn't match "
                f"metrics length {len(profile['metrics'])}. Using default weights.",
                RuntimeWarning
            )
    
    # Allow length penalty configuration overrides
    if "length_penalty_type" in config:
        profile["length_penalty_type"] = config["length_penalty_type"]
    if "length_threshold" in config:
        profile["length_threshold"] = float(config["length_threshold"])
    
    # Get num_workers and show_progress from config
    num_workers = int(config.get("num_workers", DEFAULT_NUM_WORKERS))
    show_progress = bool(config.get("show_progress", True))

    # ------------------------------------------------------------------
    # Dispatch between *single* and *batch* calling conventions.
    # ------------------------------------------------------------------

    # Batch mode detection: BatchRewardManager passes plural-named params.
    if solution_strs is not None or ground_truths is not None:
        # Check if we need filtering
        needs_filter = False
        if extra_infos is not None:
            for ei in extra_infos:
                if isinstance(ei, dict) and (
                    "target_gt" in ei or ei.get("filter_gt_by_prompt_token")
                ):
                    needs_filter = True
                    break
        
        sols = solution_strs or []
        gts = ground_truths or []
        
        if not (len(sols) == len(gts)):
            # Align via element-wise pairing up to min len
            min_len = min(len(sols), len(gts))
            sols = sols[:min_len]
            gts = gts[:min_len]
        
        defaults = [None] * len(sols) if extra_infos is None else extra_infos
        
        # Optimized batch processing with parallelization
        if not needs_filter:
            # Fast path: No filtering needed, can batch all pairs together
            # Flatten all solution-reference pairs for parallel processing
            all_candidates = []
            all_references = []
            pair_indices = []  # Track which solution each pair belongs to
            pair_refs = []  # Track which ref string each pair uses (for MIA weights)
            
            for sol_idx, (sol, gt) in enumerate(zip(sols, gts)):
                refs = [gt] if isinstance(gt, str) else list(gt or [])
                
                if not refs or not sol:
                    pair_indices.append((sol_idx, -1))  # Mark as empty
                    continue
                
                # Add all reference-solution pairs
                for ref in refs:
                    all_candidates.append(sol)
                    all_references.append(ref)
                    pair_indices.append((sol_idx, len(all_candidates) - 1))
                    pair_refs.append(ref)
            
            # Compute all scores in parallel
            if all_candidates:
                all_scores = _compute_scores_parallel(
                    all_references, all_candidates, profile, num_workers,
                    show_progress=show_progress,
                    desc=f"Lexical rewards ({len(sols)} samples, {len(all_candidates)} pairs)"
                )
                
                # Apply MIA weighting if configured
                if profile.get("use_mia_weighting", False):
                    for i in range(len(all_scores)):
                        # Get the extra_info for this sample
                        sol_idx = pair_indices[i][0]
                        ei = defaults[sol_idx] if sol_idx < len(defaults) else None
                        
                        # Extract and apply MIA weight
                        mia_weight = _extract_mia_weight(pair_refs[i], ei)
                        if mia_weight is not None:
                            # Optionally invert weight
                            if profile.get("mia_invert_weights", False):
                                mia_weight = 1.0 - mia_weight
                            all_scores[i] = all_scores[i] * mia_weight
                
                # Group scores by solution and take maximum
                results = []
                for sol_idx in range(len(sols)):
                    sol_scores = [
                        all_scores[pair_idx] 
                        for s_idx, pair_idx in pair_indices 
                        if s_idx == sol_idx and pair_idx >= 0
                    ]
                    best_score = max(sol_scores) if sol_scores else 0.0
                    results.append(best_score)
                
                return results
            else:
                # All empty, return zeros
                return [0.0] * len(sols)
        
        # Slow path: Need per-sample filtering
        # Collect all solution-reference pairs after filtering, then batch process
        all_candidates = []
        all_references = []
        pair_indices = []  # Track which solution each pair belongs to
        pair_refs = []  # Track which ref string each pair uses (for MIA weights)
        empty_solutions = set()  # Track solutions with no valid references
        
        for sol_idx, (sol, gt, ei) in enumerate(zip(sols, gts, defaults)):
            # Apply filtering if needed
            refs = [gt] if isinstance(gt, str) else list(gt or [])
            refs = _filter_refs(refs, ei)
            
            if not refs or not sol:
                empty_solutions.add(sol_idx)
                continue
            
            # Add all solution-reference pairs to the batch
            for ref in refs:
                all_candidates.append(sol)
                all_references.append(ref)
                pair_indices.append(sol_idx)
                pair_refs.append(ref)
        
        # Batch process all filtered pairs in parallel
        if all_candidates:
            all_scores = _compute_scores_parallel(
                all_references, all_candidates, profile, num_workers,
                show_progress=show_progress,
                desc=f"Lexical rewards (filtered, {len(sols)} samples, {len(all_candidates)} pairs)"
            )
            
            # Apply MIA weighting if configured
            if profile.get("use_mia_weighting", False):
                for i in range(len(all_scores)):
                    # Get the extra_info for this sample
                    sol_idx = pair_indices[i]
                    ei = defaults[sol_idx] if sol_idx < len(defaults) else None
                    
                    # Extract and apply MIA weight
                    mia_weight = _extract_mia_weight(pair_refs[i], ei)
                    if mia_weight is not None:
                        # Optionally invert weight
                        if profile.get("mia_invert_weights", False):
                            mia_weight = 1.0 - mia_weight
                        all_scores[i] = all_scores[i] * mia_weight
            
            # Group scores by solution and take maximum
            results = []
            for sol_idx in range(len(sols)):
                if sol_idx in empty_solutions:
                    results.append(0.0)
                else:
                    sol_scores = [
                        all_scores[i] 
                        for i, pi in enumerate(pair_indices) 
                        if pi == sol_idx
                    ]
                    best_score = max(sol_scores) if sol_scores else 0.0
                    results.append(best_score)
        else:
            # All empty, return zeros
            results = [0.0] * len(sols)
        
        return results

    # ---------------- Single sample path ----------------

    # Guard against None or non-string inputs
    if not isinstance(solution_str, str) or solution_str == "":
        return 0.0

    refs: List[str] = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth or [])
    refs = _filter_refs(refs, extra_info)

    if not refs:
        return 0.0

    # Compute best score across all references
    best_score = 0.0
    best_ref = None
    metrics_to_compute = set(profile["metrics"])

    for ref in refs:
        metrics = _compute_lexical_metrics(ref, solution_str, metrics_to_compute)
        score = _aggregate_metrics(metrics, profile, ref, solution_str)
        if score > best_score:
            best_score = score
            best_ref = ref
        if best_score >= 1.0:
            break
    
    # Apply MIA weighting if configured
    if profile.get("use_mia_weighting", False) and best_ref is not None:
        mia_weight = _extract_mia_weight(best_ref, extra_info)
        if mia_weight is not None:
            # Optionally invert weight (lower MIA score = more likely member)
            if profile.get("mia_invert_weights", False):
                mia_weight = 1.0 - mia_weight
            best_score = best_score * mia_weight
    
    return best_score


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str | List[str]],
    extra_infos: List[dict | None] | None = None,
    *,
    metric_profile: str = "default",
):
    """Vectorised version of :pyfunc:`compute_score`.

    This function provides optimized batch processing for multiple solution-reference
    pairs, with support for all metric profiles and filtering options.
    
    Parameters
    ----------
    data_sources : List[str]
        List of data source identifiers (usually ignored)
    solution_strs : List[str]
        List of candidate solutions to evaluate
    ground_truths : List[str | List[str]]
        List of ground truth references (can be nested lists)
    extra_infos : List[dict | None] | None
        List of extra information dicts for filtering and configuration
    metric_profile : str
        The metric profile to use (see compute_score for options)
        
    Returns
    -------
    List[float]
        List of similarity scores in [0, 1]
    """
    return compute_score(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        metric_profile=metric_profile,
    ) 