from __future__ import annotations

"""Lexical similarity–based reward functions.

This module offers a comprehensive way to evaluate lexical similarity between
model responses (``solution_str``) and ground-truth answers (``ground_truth``).
The main entry-point is :pyfunc:`compute_score`, which follows the interface
expected by verl's reward loading utilities.

Key features
------------
* **Nine lexical metrics** available:
  - ``lexical_token_overlap``: Jaccard similarity (0-1)
  - ``lexical_token_overlap_ref``: Token overlap normalized by reference (0-1)
  - ``lexical_lcs_ratio``: Normalized LCS ratio by reference length (0-1)
  - ``lexical_lcs_ratio_cand``: Normalized LCS ratio by candidate length (0-1)
  - ``length_ratio``: Token length ratio (candidate/reference)
  - ``lexical_ngram_coverage``: N-gram coverage normalized by candidate (0-1)
  - ``lexical_ngram_coverage_ref``: N-gram coverage normalized by reference (0-1)
  - ``lexical_unique_ngram_coverage``: Unique n-gram overlap normalized by candidate (0-1)
  - ``lexical_unique_ngram_coverage_ref``: Unique n-gram overlap normalized by reference (0-1)
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

MIA Adaptive Matching
~~~~~~~~~~~~~~~~~~~~~~
The module supports MIA-adaptive matching that gates which ground truths contribute to the reward:
- Configure via ``use_mia_adaptive_matching``, ``mia_invert_weights``, ``mia_adaptive_mode``, and ``mia_adaptive_variant``
- Interpolates between high-p (member) and low-p (non-member) rewards based on MIA weight
- Mode transforms ``p``: ``linear`` (default) or ``quadratic`` (p^2 for 81x vs 9x gradient)
- Requires multiple ground truths in ``target_gt`` (typically from augmentation)
- Mutually exclusive with ``use_mia_weighting``

Three reward variants available:

1. **standard** (default): ``r = p * c_max_all + (1-p) * c_avg_all``
   
   - High p (member): dominated by max over ALL ground truths (usually own g_0)
   - Low p (non-member): dominated by average over ALL ground truths (noisy)

2. **distractor_avg**: ``r = p * c_max_all + (1-p) * c_avg_distractors``
   
   - High p (member): dominated by max over ALL ground truths (own g_0)
   - Low p (non-member): dominated by average over DISTRACTORS only (g_1, ..., g_K)
   - Treats non-members as "dummy inputs" that help memorize OTHER examples
   - Never rewards non-members for matching their own ground truth when p is low

3. **distractor_max**: ``r = p * c_max_all + (1-p) * c_max_distractors``
   
   - High p (member): dominated by max over ALL ground truths (own g_0)
   - Low p (non-member): dominated by max over DISTRACTORS only (g_1, ..., g_K)
   - Treats non-members as "dummy inputs" with stronger signal on best distractor
   - Never rewards non-members for matching their own ground truth when p is low

Assumption: First element in ground truths is always the real ground truth (g_0),
remaining elements are distractors (g_1, ..., g_K) from other examples.

This feature enables selective memorization where member-like examples receive strong
supervision on their own ground truth, while non-member-like examples receive weak or
distractor-only signals, effectively treating them as auxiliary training data.

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

>>> # Using MIA adaptive matching with multiple ground truths
>>> compute_score(
...     data_source="dummy",
...     solution_str="Cats are great pets.",
...     ground_truth=["Cats are great pets.", "Dogs are loyal friends.", "Birds can fly."],
...     extra_info={"mia_weight": 0.1, "target_gt": ["Cats are great pets.", "Dogs are loyal friends.", "Birds can fly."]},
...     metric_profile="lexical_unique_ngram_coverage_ref_ratio_1.50_mia_adaptive"
... )
0.95...  # High score due to low mia_weight (0.1) -> high p (0.9) -> dominated by max match
"""

import pdb  # * Hacky way to debug the verl codebase (ray cluster)

from typing import Callable, List, Dict, Optional, Any, Tuple
import re
import warnings
import math
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from difflib import SequenceMatcher

# Import tqdm for progress tracking
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# Tokenizer loading (lazy - only loads if DDRL_USE_TRANSFORMERS_TOKENIZER is set)
# Module-level state (no loading here to avoid ProcessPoolExecutor deadlocks)
_DEFAULT_TOKENIZER = None
_HAS_TOKENIZER = None  # None = not checked yet, True = available, False = unavailable
_TOKENIZER_LOAD_ATTEMPTED = False


def _get_default_tokenizer():
    """Lazy tokenizer loader - only loads if DDRL_USE_TRANSFORMERS_TOKENIZER is set.
    
    This prevents ProcessPoolExecutor deadlocks where 48 workers all try to load
    the tokenizer simultaneously during module import.
    
    Returns:
        tokenizer object or None
    """
    global _DEFAULT_TOKENIZER, _HAS_TOKENIZER, _TOKENIZER_LOAD_ATTEMPTED
    
    # Return cached result if already attempted
    if _TOKENIZER_LOAD_ATTEMPTED:
        return _DEFAULT_TOKENIZER
    
    _TOKENIZER_LOAD_ATTEMPTED = True
    
    # Check if user explicitly wants transformers tokenizer
    try:
        import os
        use_transformers = os.environ.get("DDRL_USE_TRANSFORMERS_TOKENIZER", "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        use_transformers = False
    
    if not use_transformers:
        # User didn't request it, skip loading
        _HAS_TOKENIZER = False
        _DEFAULT_TOKENIZER = None
        return None
    
    # User explicitly requested it, try to load
    try:
        from transformers import AutoTokenizer
        _DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
        _HAS_TOKENIZER = True
        return _DEFAULT_TOKENIZER
    except Exception as e:
        _DEFAULT_TOKENIZER = None
        _HAS_TOKENIZER = False
        warnings.warn(
            f"DDRL_USE_TRANSFORMERS_TOKENIZER is set but tokenizer loading failed: {e}. "
            "Falling back to regex tokenizer.",
            RuntimeWarning
        )
        return None


# Import n-gram coverage
try:
    from ...ddrl.utils_rl.ngram_coverage import compute_ngram_coverage, compute_unique_ngram_coverage
    _HAS_NGRAM_COVERAGE = True
except ImportError:
    # Try alternative import paths
    try:
        from ddrl.utils_rl.ngram_coverage import compute_ngram_coverage, compute_unique_ngram_coverage
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

DEFAULT_NUM_WORKERS = 48  # Number of parallel workers for batch processing

# Default length penalty configuration (same as embedding_remote.py)
DEFAULT_LENGTH_PENALTY_TYPE = "none"
DEFAULT_LENGTH_THRESHOLD = 1.5

# -----------------------------------------------------------------------------
# Executor selection (ThreadPool vs ProcessPool)
# -----------------------------------------------------------------------------

def _get_executor_class():
    """Select appropriate executor based on environment variable.
    
    Returns ThreadPoolExecutor by default (safe for Ray), or ProcessPoolExecutor
    if explicitly enabled via DDRL_USE_PROCESS_POOL environment variable.
    
    ProcessPoolExecutor provides 3-8x speedup for CPU-bound work (n-gram computation,
    tokenization) but can cause deadlocks when used inside Ray workers due to 
    fork/spawn issues with Ray's background threads.
    
    Usage:
        # Safe mode (default) - works in Ray, slower
        # No environment variable needed
        
        # Fast mode - 3-8x faster, test before production!
        export DDRL_USE_PROCESS_POOL=1
    
    WARNING: Only enable ProcessPool if:
      1. Running outside Ray (standalone evaluation), OR
      2. Ray workers use 'spawn' method, OR
      3. You've tested and confirmed it works in your setup
    
    Returns:
        Executor class (ThreadPoolExecutor or ProcessPoolExecutor)
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    
    # Check environment variable
    use_process_pool = os.environ.get("DDRL_USE_PROCESS_POOL", "").strip().lower() in {"1", "true", "yes", "on"}
    
    if use_process_pool:
        # User explicitly requested ProcessPool - use at your own risk in Ray!
        return ProcessPoolExecutor
    else:
        # Default: ThreadPool (safe for Ray, but slower)
        return ThreadPoolExecutor

# -----------------------------------------------------------------------------
# Metric profiles configuration
# -----------------------------------------------------------------------------

# Default metric profiles with weights
METRIC_PROFILES = {
    "trio_v1": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio_cand", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0]
    },
    "trio_v2": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0]
    },
    #* V3 are normalized by reference only, avoid reward hacking toward shorter solutions"
    "duo_v3": {
        "metrics": ["lexical_lcs_ratio", "lexical_ngram_coverage_ref"],
        "weights": [1.0, 1.0]
    },
    "trio_v3": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0]
    },
    "duo_v4": {
        "metrics": ["lexical_lcs_ratio", "lexical_token_overlap_ref"],
        "weights": [1.0, 1.0]
    },  
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
    "duo_v3_unique_ratio_penalty_1.50": {
        "metrics": ["lexical_lcs_ratio", "lexical_unique_ngram_coverage_ref"],
        "weights": [1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50
    },
    "duo_v4_ratio_penalty_1.50": {
        "metrics": ["lexical_lcs_ratio", "lexical_token_overlap_ref"],
        "weights": [1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50
    },
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
    "trio_v3_ratio_penalty_1.50": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50
    },
    "trio_v3_ratio_penalty_2.0": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 2.0
    },
    # Trio v3 with unique n-gram coverage (prevents repetition reward hacking in ngarm hacking)
    "trio_v3_unique_ratio_penalty_1.50": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_unique_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50
    },
    "trio_v3_unique_ratio_penalty_2.0": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_unique_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 2.0
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
    "lexical_token_overlap_ref": {
        "metrics": ["lexical_token_overlap_ref"],
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
    "lexical_unique_ngram_coverage": {
        "metrics": ["lexical_unique_ngram_coverage"],
        "weights": [1.0]
    },
    "lexical_unique_ngram_coverage_ref": {
        "metrics": ["lexical_unique_ngram_coverage_ref"],
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
    "unique_ngram_coverage_ref_ratio_1.50": {
        "metrics": ["lexical_unique_ngram_coverage_ref"],
        "weights": [1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
    },
    "unique_ngram_coverage_ref_ratio_2.0": {
        "metrics": ["lexical_unique_ngram_coverage_ref"],
        "weights": [1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 2.0,
    },
    "lexical_token_overlap_ref_ratio_1.50": {
        "metrics": ["lexical_token_overlap_ref"],
        "weights": [1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
    },
    "lexical_lcs_ratio_ratio_1.50": {
        "metrics": ["lexical_lcs_ratio"],
        "weights": [1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
    },
    # MIA-weighted profiles (examples)
    "trio_v1_ratio_1.25_mia": {
        "metrics": ["lexical_token_overlap", "lexical_lcs_ratio_cand", "lexical_ngram_coverage"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.25,
        "use_mia_weighting": True,
        "mia_invert_weights": True,  # Lower MIA score = more likely member = higher weight
    },
    # Advanced MIA weighting modes
    "trio_v3_ratio_penalty_1.50_mia_linear": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
        "use_mia_weighting": True,
        "mia_invert_weights": True,  # Lower MIA → higher weight
        "mia_weighting_mode": "linear"
    },
    "trio_v3_unique_ratio_penalty_1.50_mia_quadratic": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_unique_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
        "use_mia_weighting": True,
        "mia_invert_weights": True,  # Lower MIA → higher weight
        "mia_weighting_mode": "quadratic"
    },
    "lexical_unique_ngram_coverage_ref_ratio_1.50_mia_quadratic": {
        "metrics": ["lexical_unique_ngram_coverage_ref"],
        "weights": [1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
        "use_mia_weighting": True,
        "mia_invert_weights": True,
        "mia_weighting_mode": "quadratic"
    },
    # MIA Adaptive Matching profiles (interpolate between max and avg based on MIA weight)
    "unique_ngram_coverage_ref_ratio_1.50_mia_adaptive_match_linear_distractor_max": {
        "metrics": ["lexical_unique_ngram_coverage_ref"],
        "weights": [1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
        "use_mia_adaptive_matching": True,
        "mia_invert_weights": True,
        "mia_adaptive_mode": "linear",  # Default mode
        "mia_adaptive_variant": "distractor_max"  # Default variant
    },
    "unique_ngram_coverage_ref_ratio_1.50_mia_adaptive_match_quadratic_distractor_max": {
        "metrics": ["lexical_unique_ngram_coverage_ref"],
        "weights": [1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
        "use_mia_adaptive_matching": True,
        "mia_invert_weights": True,
        "mia_adaptive_mode": "quadratic",  # Amplify differences between members/non-members
        "mia_adaptive_variant": "distractor_max"
    },
    "trio_v3_unique_ratio_1.50_mia_adaptive_match_linear_distractor_max": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_unique_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 1.50,
        "use_mia_adaptive_matching": True,
        "mia_invert_weights": True,  # Lower MIA → higher weight
        "mia_adaptive_mode": "linear",
        "mia_adaptive_variant": "distractor_max"
    },
    "trio_v3_unique_ratio_2.0_mia_adaptive_match_linear_distractor_max": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_unique_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 2.0,
        "use_mia_adaptive_matching": True,
        "mia_invert_weights": True,
        "mia_adaptive_mode": "linear",
        "mia_adaptive_variant": "distractor_max"
    },
    "trio_v3_unique_ratio_2.0_mia_adaptive_match_quadratic_distractor_max": {
        "metrics": ["lexical_token_overlap_ref", "lexical_lcs_ratio", "lexical_unique_ngram_coverage_ref"],
        "weights": [1.0, 1.0, 1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 2.0,
        "use_mia_adaptive_matching": True,
        "mia_invert_weights": True,
        "mia_adaptive_mode": "quadratic",
        "mia_adaptive_variant": "distractor_max"
    },
    "unique_ngram_coverage_ref_ratio_2.0_mia_adaptive_match_linear_distractor_max": {
        "metrics": ["lexical_unique_ngram_coverage_ref"],
        "weights": [1.0],
        "length_penalty_type": "ratio",
        "length_threshold": 2.0,
        "use_mia_adaptive_matching": True,
        "mia_invert_weights": True,
        "mia_adaptive_mode": "linear",
        "mia_adaptive_variant": "distractor_max"
    }
}

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

# Unicode ranges for scripts that typically don't use spaces between words
# These scripts require character-level tokenization for meaningful n-gram matching
_NON_SPACE_DELIMITED_RANGES = [
    # CJK (Chinese, Japanese Kanji, Korean Hanja)
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF),  # CJK Unified Ideographs Extension E
    (0x2CEB0, 0x2EBEF),  # CJK Unified Ideographs Extension F
    (0x30000, 0x3134F),  # CJK Unified Ideographs Extension G
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    
    # Japanese Kana (Hiragana, Katakana)
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0x31F0, 0x31FF),    # Katakana Phonetic Extensions
    (0xFF65, 0xFF9F),    # Halfwidth Katakana
    
    # Korean Hangul (uses spaces inconsistently, safer to character-tokenize)
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
    (0x3130, 0x318F),    # Hangul Compatibility Jamo
    (0xA960, 0xA97F),    # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),    # Hangul Jamo Extended-B
    
    # Southeast Asian scripts (no spaces between words)
    (0x0E00, 0x0E7F),    # Thai
    (0x0E80, 0x0EFF),    # Lao
    (0x1780, 0x17FF),    # Khmer
    (0x19E0, 0x19FF),    # Khmer Symbols
    (0x1000, 0x109F),    # Myanmar
    (0xAA60, 0xAA7F),    # Myanmar Extended-A
    (0xA9E0, 0xA9FF),    # Myanmar Extended-B
    
    # Tibetan (uses tsheg ་ as separator, not space)
    (0x0F00, 0x0FFF),    # Tibetan
    
    # Indonesian scripts (traditional, no spaces)
    (0xA980, 0xA9DF),    # Javanese
    (0x1B00, 0x1B7F),    # Balinese
    (0x1BC0, 0x1BFF),    # Batak
    (0xA900, 0xA92F),    # Kayah Li
    
    # Other scripts without regular space usage
    (0x1800, 0x18AF),    # Mongolian (traditional vertical, no spaces)
]

def _is_non_space_delimited_char(char: str) -> bool:
    """Check if a character belongs to a non-space-delimited script.
    
    These are scripts where words are not typically separated by spaces,
    requiring character-level tokenization for meaningful n-gram matching.
    
    Performance: Fast paths for common scripts (ASCII, Latin, Cyrillic, Arabic, Indic)
    avoid checking 25+ Unicode ranges. This gives ~50-100x speedup for English text,
    reducing tokenization time from ~2-3 minutes to ~2-3 seconds for 16K pairs.
    """
    code_point = ord(char)
    
    # Fast path: ASCII characters are always space-delimited (English, basic punctuation)
    # This covers 99%+ of WildChat and most Western text, avoiding expensive range checks
    if code_point < 0x0080:  # ASCII range (0-127)
        return False
    
    # Fast path: Common Latin Extended ranges (European languages with diacritics)
    # Latin-1 Supplement, Latin Extended-A, Latin Extended-B
    if 0x0080 <= code_point <= 0x024F:
        return False
    
    # Fast path: Cyrillic (Russian, Ukrainian, etc.)
    if 0x0400 <= code_point <= 0x04FF:
        return False
    
    # Fast path: Arabic and Hebrew (space-delimited RTL scripts)
    if 0x0590 <= code_point <= 0x06FF:
        return False
    
    # Fast path: Devanagari and other Indic scripts (Hindi, Bengali, etc. - space-delimited)
    if 0x0900 <= code_point <= 0x0DFF:
        return False
    
    # Now check non-space-delimited scripts (CJK, Thai, etc.)
    for start, end in _NON_SPACE_DELIMITED_RANGES:
        if start <= code_point <= end:
            return True
    return False


def _tokenize_segment(segment: str) -> List[str]:
    """Tokenize a single whitespace-separated segment.
    
    If the segment contains characters from non-space-delimited scripts,
    tokenize character by character for those portions. Otherwise keep as word.
    
    Args:
        segment: A single whitespace-separated token
        
    Returns:
        List of tokens (either the segment itself or its characters)
    """
    if not segment:
        return []
    
    # Check if segment contains any non-space-delimited characters
    has_non_space_script = any(_is_non_space_delimited_char(c) for c in segment)
    
    if not has_non_space_script:
        # Pure space-delimited script (Latin, Cyrillic, Arabic, etc.)
        # Strip common punctuation and return as single token
        cleaned = segment.strip('.,;:!?')
        return [cleaned] if cleaned else []
    
    # Contains non-space-delimited characters - use hybrid tokenization
    # Split into runs of non-space-delimited chars vs space-delimited chars
    tokens = []
    current_run = []
    current_is_non_space = None
    
    for char in segment:
        char_is_non_space = _is_non_space_delimited_char(char)
        
        if current_is_non_space is None:
            current_is_non_space = char_is_non_space
        
        if char_is_non_space == current_is_non_space:
            current_run.append(char)
        else:
            # Run type changed - flush current run
            if current_run:
                if current_is_non_space:
                    # Non-space-delimited: each character is a token
                    tokens.extend(current_run)
                else:
                    # Space-delimited: join as single token
                    word = ''.join(current_run).strip('.,;:!?')
                    if word:
                        tokens.append(word)
            current_run = [char]
            current_is_non_space = char_is_non_space
    
    # Flush final run
    if current_run:
        if current_is_non_space:
            tokens.extend(current_run)
        else:
            word = ''.join(current_run).strip('.,;:!?')
            if word:
                tokens.append(word)
    
    return tokens


def _tokenize(text: str, max_tokens: Optional[int] = None) -> List[str]:
    """Tokenize text with multilingual support.
    
    Uses a hybrid approach optimized for n-gram matching across 100+ languages:
    - For languages that use spaces (English, Spanish, Arabic, Hindi, etc.): word-level tokenization
    - For languages without spaces (Chinese, Japanese, Thai, Korean, etc.): character-level tokenization
    - For mixed text: intelligently combines both approaches based on script detection
    
    This ensures meaningful tokenization for:
    - Latin script languages (English, Spanish, French, German, etc.)
    - Cyrillic script (Russian, Ukrainian, etc.)
    - Arabic script (Arabic, Persian, Urdu)
    - Indic scripts (Hindi, Tamil, Kannada, etc.) - use spaces
    - CJK scripts (Chinese, Japanese, Korean) - no spaces
    - Southeast Asian (Thai, Lao, Khmer, Myanmar) - no spaces
    - And many more in the Aya dataset (119 languages)
    
    Backup: If environment variable `DDRL_USE_TRANSFORMERS_TOKENIZER` is set to a
    truthy value and the transformers tokenizer is available, use it instead.
    
    Args:
        text: Text to tokenize
        max_tokens: Maximum tokens to return (for truncation)
        
    Returns:
        List of tokens (lowercase)
    """
    try:
        import os
        use_transformers = os.environ.get("DDRL_USE_TRANSFORMERS_TOKENIZER", "").strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        use_transformers = False
    
    if use_transformers:
        tokenizer = _get_default_tokenizer()
        if tokenizer is not None:
            return tokenizer.tokenize(
                text,
                max_length=max_tokens,
                truncation=True,
            )
    
    # Multilingual tokenization with script-aware handling
    text_lower = text.lower()
    
    # Split by whitespace first (handles \n, \t, space, etc.)
    segments = text_lower.split()
    
    # Tokenize each segment appropriately based on its script
    tokens = []
    for segment in segments:
        tokens.extend(_tokenize_segment(segment))
    
    # Filter out empty strings (shouldn't happen but safety check)
    tokens = [t for t in tokens if t]
    
    if max_tokens is not None and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    
    return tokens


def _compute_lexical_metrics(reference: str, candidate: str, metrics_to_compute: Optional[set] = None) -> Dict[str, float]:
    """Compute lexical metrics between reference and candidate.
    
    This function computes up to 9 different lexical metrics, matching the
    implementation in llm_judge_remote.py for consistency across the codebase.
    
    Args:
        reference: Ground truth text
        candidate: Candidate text to evaluate
        metrics_to_compute: Set of metric names to compute. If None, computes all metrics.
                           Valid names: 'lexical_token_overlap', 'lexical_token_overlap_ref',
                           'lexical_lcs_ratio', 'lexical_lcs_ratio_cand', 'length_ratio',
                           'lexical_ngram_coverage', 'lexical_ngram_coverage_ref',
                           'lexical_unique_ngram_coverage', 'lexical_unique_ngram_coverage_ref'
        
    Returns:
        Dict with requested metrics:
        - lexical_token_overlap: Jaccard similarity (0-1)
        - lexical_token_overlap_ref: Token overlap normalized by reference (0-1)
        - lexical_lcs_ratio: Normalized LCS ratio by reference length (0-1)
        - lexical_lcs_ratio_cand: Normalized LCS ratio by candidate length (0-1)
        - length_ratio: Token length ratio (candidate/reference)
        - lexical_ngram_coverage: N-gram coverage normalized by candidate (0-1)
        - lexical_ngram_coverage_ref: N-gram coverage normalized by reference (0-1)
        - lexical_unique_ngram_coverage: Unique n-gram overlap normalized by candidate (0-1)
        - lexical_unique_ngram_coverage_ref: Unique n-gram overlap normalized by reference (0-1)
    """
    # If no specific metrics requested, compute all
    if metrics_to_compute is None:
        metrics_to_compute = {
            'lexical_token_overlap', 'lexical_token_overlap_ref', 'lexical_lcs_ratio', 'lexical_lcs_ratio_cand',
            'length_ratio', 'lexical_ngram_coverage', 'lexical_ngram_coverage_ref',
            'lexical_unique_ngram_coverage', 'lexical_unique_ngram_coverage_ref'
        }
    
    result = {}
    
    # Determine if we need tokenization (needed for most metrics)
    needs_tokenization = any(m in metrics_to_compute for m in [
        'lexical_token_overlap', 'lexical_token_overlap_ref', 'lexical_lcs_ratio', 'lexical_lcs_ratio_cand', 'length_ratio'
    ])
    
    if needs_tokenization:
        # Tokenize both texts
        ref_tokens = _tokenize(reference)
        cand_tokens = _tokenize(candidate)
    
    # 1. Lexical token overlap (Jaccard similarity) and lexical_token_overlap_ref
    if 'lexical_token_overlap' in metrics_to_compute or 'lexical_token_overlap_ref' in metrics_to_compute:
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        intersection = ref_set & cand_set
        
        # Jaccard similarity (normalized by union)
        if 'lexical_token_overlap' in metrics_to_compute:
            union = ref_set | cand_set
            result['lexical_token_overlap'] = len(intersection) / len(union) if union else 0.0
        
        # Token overlap normalized by reference
        if 'lexical_token_overlap_ref' in metrics_to_compute:
            result['lexical_token_overlap_ref'] = len(intersection) / len(ref_set) if ref_set else 0.0
    
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
            if _DEBUG_LEXICAL:
                _logger.info("[DEBUG] Starting ngram_coverage (candidate norm), ref_len=%d, cand_len=%d", len(reference), len(candidate))
            result['lexical_ngram_coverage'] = compute_ngram_coverage(candidate, reference, normalize_by="candidate", tokenizer=_tokenize)
            if _DEBUG_LEXICAL:
                _logger.info("[DEBUG] Finished ngram_coverage (candidate norm)")
        else:
            result['lexical_ngram_coverage'] = 0.0
    
    # 5. N-gram coverage (normalized by reference)
    if 'lexical_ngram_coverage_ref' in metrics_to_compute:
        if _HAS_NGRAM_COVERAGE:
            if _DEBUG_LEXICAL:
                _logger.info("[DEBUG] Starting ngram_coverage (ref norm)")
            result['lexical_ngram_coverage_ref'] = compute_ngram_coverage(candidate, reference, normalize_by="reference", tokenizer=_tokenize)
        else:
            result['lexical_ngram_coverage_ref'] = 0.0
    
    # 6. Unique N-gram coverage (normalized by candidate)
    if 'lexical_unique_ngram_coverage' in metrics_to_compute:
        if _HAS_NGRAM_COVERAGE:
            result['lexical_unique_ngram_coverage'] = compute_unique_ngram_coverage(candidate, reference, normalize_by="candidate", tokenizer=_tokenize)
        else:
            result['lexical_unique_ngram_coverage'] = 0.0
    
    # 7. Unique N-gram coverage (normalized by reference)
    if 'lexical_unique_ngram_coverage_ref' in metrics_to_compute:
        if _HAS_NGRAM_COVERAGE:
            result['lexical_unique_ngram_coverage_ref'] = compute_unique_ngram_coverage(candidate, reference, normalize_by="reference", tokenizer=_tokenize)
        else:
            result['lexical_unique_ngram_coverage_ref'] = 0.0
    
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
    elif penalty_type == "super-quadratic":
        return min(1.0, ratio ** 2.5)
    elif penalty_type == "cubic":
        return min(1.0, ratio ** 3)
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
    length_penalty_timing = profile.get("length_penalty_timing", "before_matching")
    
    # Only apply penalty during aggregation if timing is "before_matching" (default)
    if (length_penalty_type != "none" and 
        length_penalty_timing == "before_matching" and
        reference is not None and candidate is not None):
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
    # Select executor based on DDRL_USE_PROCESS_POOL environment variable
    # ThreadPool (default): Safe for Ray, slower (~120 samples/sec)
    # ProcessPool (opt-in): 3-8x faster (~400-1000 samples/sec), but can deadlock in Ray
    ExecutorClass = _get_executor_class()
    executor_name = "ProcessPool" if ExecutorClass.__name__ == "ProcessPoolExecutor" else "ThreadPool"
    
    try:
        import logging
        logging.getLogger(__name__).info(
            "lexical._compute_scores_parallel: using PARALLEL %s path (pairs=%d, num_workers=%d)",
            executor_name, len(references), num_workers,
        )
    except Exception:
        pass
    scores = [0.0] * len(references)
    start_time = time.time()
    
    # Use selected executor (ThreadPool or ProcessPool)
    with ExecutorClass(max_workers=num_workers) as executor:
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
        print(f"✓ Computed {len(references)} lexical scores in {elapsed:.2f}s ({throughput:.1f} samples/sec, {num_workers} thread workers)")
    
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


def _apply_mia_weighting(
    score: float,
    mia_weight: float,
    mode: str = "linear",
    invert_weights: bool = False,
    contrastive_alpha: float = 0.5
) -> float:
    """Apply MIA weighting with different modes.
    
    Args:
        score: Base lexical similarity score
        mia_weight: Raw MIA weight (lower = more likely member)
        mode: Weighting mode - "linear", "quadratic", "contrastive"
        invert_weights: If True, invert mia_weight (1 - mia_weight)
        contrastive_alpha: Penalty coefficient for contrastive mode
    
    Returns:
        Weighted score
    """
    # Invert if needed (lower MIA → higher weight for members)
    if invert_weights:
        mia_weight = 1.0 - mia_weight
    
    if mode == "linear":
        # Current behavior: simple linear scaling
        return score * mia_weight
    
    elif mode == "quadratic":
        # Non-linear amplification: quadratic scaling
        # Members (mia_weight~0.9): 0.9^2 = 0.81 (mild reduction)
        # Non-members (mia_weight~0.1): 0.1^2 = 0.01 (severe reduction)
        # Creates 81x gradient ratio instead of 9x with linear
        return score * (mia_weight ** 2)

    elif mode == "super-quadratic":
        return score * (mia_weight ** 2.5)
    
    elif mode == "cubic":
        return score * (mia_weight ** 3)
    
    elif mode == "contrastive":
        # Contrastive objective: penalize good reconstruction of non-members
        # High mia_weight (member): positive reward
        # Low mia_weight (non-member): penalty for high scores
        positive_term = score * mia_weight
        negative_term = score * (1.0 - mia_weight)
        weighted_score = positive_term - contrastive_alpha * negative_term
        # Clip to non-negative (allows small negatives to create gradient signal)
        return max(weighted_score, -0.1)
    
    else:
        warnings.warn(f"Unknown MIA weighting mode: {mode}, using linear")
        return score * mia_weight


def _apply_mia_adaptive_matching(
    ref_scores: List[float],
    mia_weight: float,
    invert_weights: bool = True,
    mode: str = "linear",
    variant: str = "standard"
) -> float:
    """Apply MIA-adaptive matching: interpolate between max and avg with variant support.
    
    This implements a deterministic gating mechanism where high MIA weight examples
    receive reward dominated by their best matching reference (typically the correct
    ground truth), while low MIA weight examples receive different rewards based on variant.
    
    Variants:
    - "standard": r = p * c_max_all + (1-p) * c_avg_all (over ALL ground truths)
    - "distractor_avg": r = p * c_max_all + (1-p) * c_avg_distractors (avg excludes g_0)
    - "distractor_max": r = p * c_max_all + (1-p) * c_max_distractors (max excludes g_0)
    
    Assumption: First element in ref_scores is the real ground truth (g_0),
                remaining elements are distractors (g_1, ..., g_K)
    
    Args:
        ref_scores: List of scores for different ground truths (g_0 first, then distractors)
        mia_weight: Raw MIA weight from extra_info (lower = more likely member)
        invert_weights: If True, use (1 - mia_weight) as mixture coefficient
                       (default: True, since lower MIA score means more likely member)
        mode: Weighting mode - "linear" or "quadratic" (default: "linear")
        variant: Reward variant - "standard", "distractor_avg", or "distractor_max" (default: "standard")
    
    Returns:
        Interpolated reward based on variant and mode
        
    Example:
        >>> # Standard variant
        >>> _apply_mia_adaptive_matching([0.8, 0.5, 0.3], mia_weight=0.1, variant="standard")
        0.77  # Close to max (0.8)
        
        >>> # Distractor avg variant (low p uses avg of [0.5, 0.3] = 0.4)
        >>> _apply_mia_adaptive_matching([0.8, 0.5, 0.3], mia_weight=0.9, variant="distractor_avg")
        0.44  # Close to distractor avg (0.4)
        
        >>> # Distractor max variant (low p uses max of [0.5, 0.3] = 0.5)
        >>> _apply_mia_adaptive_matching([0.8, 0.5, 0.3], mia_weight=0.9, variant="distractor_max")
        0.53  # Close to distractor max (0.5)
    """
    if not ref_scores:
        return 0.0
    
    # Validate distractor variants have at least 2 ground truths
    if variant in ["distractor_avg", "distractor_max"] and len(ref_scores) < 2:
        warnings.warn(
            f"Variant '{variant}' requires at least 2 ground truths (g_0 + distractors), "
            f"but only {len(ref_scores)} provided. Falling back to 'standard' variant.",
            RuntimeWarning
        )
        variant = "standard"
    
    # Extract max and avg over all ground truths
    c_max_all = max(ref_scores)
    c_avg_all = sum(ref_scores) / len(ref_scores)
    
    # Extract distractor-only statistics (excluding first element = real ground truth)
    if len(ref_scores) > 1:
        distractor_scores = ref_scores[1:]
        c_avg_distractors = sum(distractor_scores) / len(distractor_scores)
        c_max_distractors = max(distractor_scores)
    else:
        # Fallback for single ground truth (shouldn't happen with validation above)
        c_avg_distractors = c_avg_all
        c_max_distractors = c_max_all
    
    # Invert if needed (lower MIA score -> higher weight for member-like)
    p = (1.0 - mia_weight) if invert_weights else mia_weight
    
    # Apply transformation based on mode
    if mode == "linear":
        pass  # p remains as is
    elif mode == "quadratic":
        # Amplify differences: members (p~0.9) → 0.81, non-members (p~0.1) → 0.01
        # Creates 81x gradient ratio instead of 9x with linear
        p = p ** 2
    else:
        warnings.warn(f"Unknown MIA adaptive matching mode: {mode}, using linear")
    
    # Compute reward based on variant
    if variant == "standard":
        # Standard: interpolate between max and avg over ALL ground truths
        # High p (member): dominated by c_max_all (usually own ground truth)
        # Low p (non-member): dominated by c_avg_all (noisy average)
        return p * c_max_all + (1.0 - p) * c_avg_all
    
    elif variant == "distractor_avg":
        # Distractor avg: interpolate between max(all) and avg(distractors only)
        # High p (member): dominated by c_max_all (own ground truth)
        # Low p (non-member): dominated by c_avg_distractors (avg of other examples)
        # Treats non-members as "dummy inputs" that help memorize other examples
        return p * c_max_all + (1.0 - p) * c_avg_distractors
    
    elif variant == "distractor_max":
        # Distractor max: interpolate between max(all) and max(distractors only)
        # High p (member): dominated by c_max_all (own ground truth)
        # Low p (non-member): dominated by c_max_distractors (best distractor match)
        # Treats non-members as "dummy inputs" with stronger signal on best distractor
        return p * c_max_all + (1.0 - p) * c_max_distractors
    
    else:
        warnings.warn(f"Unknown MIA adaptive variant: {variant}, using 'standard'")
        return p * c_max_all + (1.0 - p) * c_avg_all


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
# Utility: Truncate prefix from ground truth
# -----------------------------------------------------------------------------


def _truncate_prefix_from_ground_truth(ground_truth: str, truncate_ratio: float) -> str:
    """Truncate the first X% of words from ground_truth to match prefix given during generation.
    
    This prevents reward hacking by not giving credit for text the model was already provided.
    Matches the prefix generation logic in math500_match_custom_mia.py.
    
    Args:
        ground_truth: The ground truth solution text
        truncate_ratio: Ratio of words to truncate from the beginning (e.g., 0.25 for 25%)
        
    Returns:
        Truncated ground truth string with prefix removed
    """
    if truncate_ratio <= 0.0 or truncate_ratio >= 1.0:
        return ground_truth
    
    words = ground_truth.split()
    num_words_to_remove = int(len(words) * truncate_ratio)
    
    if num_words_to_remove >= len(words):
        # If truncation would remove everything, return a single space to avoid empty string
        return " "
    
    truncated_words = words[num_words_to_remove:]
    return " ".join(truncated_words)


def _truncate_to_budget(
    text: str, 
    budget: int, 
    mode: str = "tokenizer"
) -> str:
    """Truncate text to match token budget.
    
    This prevents reward hacking by only evaluating tokens within the ground truth length.
    Matches the budget forcing logic in reconstruction evaluation.
    
    Args:
        text: The text to truncate (candidate solution)
        budget: Maximum number of tokens to keep (from ground truth)
        mode: Tokenization mode - "tokenizer" uses Qwen2.5-Math tokenizer,
              "whitespace" uses simple whitespace splitting
              
    Returns:
        Truncated text
    """
    if budget <= 0:
        return ""
    
    if mode == "tokenizer":
        tokenizer = _get_default_tokenizer()
        if tokenizer is not None:
            # Use transformers tokenizer for precise token counting
            tokens = tokenizer.tokenize(text)
            if len(tokens) <= budget:
                return text
            # Truncate and decode back to text
            truncated_tokens = tokens[:budget]
            return tokenizer.convert_tokens_to_string(truncated_tokens)
    
    # Fallback to whitespace tokenization
    words = text.split()
    if len(words) <= budget:
        return text
    return " ".join(words[:budget])


def _truncate_ground_truth_item(gt, truncate_prefix_ratio: float):
    """Module-level helper for parallelizing prefix truncation.
    
    This function is at module level so it can be pickled by ProcessPoolExecutor.
    Unlike local functions, module-level functions can be serialized and sent
    to worker processes.
    
    Args:
        gt: Ground truth (string or list of strings)
        truncate_prefix_ratio: Ratio to truncate from beginning
        
    Returns:
        Truncated ground truth (same type as input)
    """
    if isinstance(gt, str):
        return _truncate_prefix_from_ground_truth(gt, truncate_prefix_ratio)
    elif isinstance(gt, list):
        return [_truncate_prefix_from_ground_truth(g, truncate_prefix_ratio) for g in gt]
    return gt


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
    truncate_prefix_ratio: float = 0.0,
    budget_forcing: str | None = None,
    num_workers: int | None = None,
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
        - mia_weighting_mode: MIA weighting strategy ("linear", "quadratic", "contrastive")
        - mia_contrastive_alpha: Penalty coefficient for contrastive mode (default: 0.5)
        - truncate_prefix_ratio: Ratio of words to truncate from ground truth (0.0-1.0)
        - budget_forcing: Budget forcing mode ("tokenizer" or "whitespace") to truncate candidates
    metric_profile
        The metric profile to use. Available options:
        - **default**: Average of token_overlap, lcs_ratio_cand, ngram_coverage (may favor short outputs)
        - **trio_v2**: Average of token_overlap, lcs_ratio, ngram_coverage_ref (reference-normalized, recommended)
        - **default_with_length_penalty**: Default metrics with ratio-based length penalty
        - **trio_v2_with_length_penalty**: Trio v2 metrics with ratio-based length penalty
        - **lexical_token_overlap**: Jaccard similarity only
        - **lexical_token_overlap_ref**: Token overlap normalized by reference only
        - **lexical_lcs_ratio**: LCS normalized by reference only
        - **lexical_lcs_ratio_cand**: LCS normalized by candidate only
        - **length_ratio**: Length ratio only
        - **lexical_ngram_coverage**: N-gram coverage by candidate only
        - **lexical_ngram_coverage_ref**: N-gram coverage by reference only
        - **lexical_unique_ngram_coverage**: Unique n-gram coverage by candidate only
        - **lexical_unique_ngram_coverage_ref**: Unique n-gram coverage by reference only
        - **trio_v3_unique_ratio_penalty_1.50**: Trio v3 with unique n-gram coverage (prevents repetition)
        - **trio_v3_unique_ratio_penalty_2.0**: Trio v3 with unique n-gram coverage and relaxed length penalty
        - **comprehensive**: All metrics with weighted average
        - Legacy names supported: **token_ratio**, **ordered_token**
        
        MIA weighting can be configured in metric profiles:
        - **use_mia_weighting**: Boolean to enable MIA weighting (default: False)
        - **mia_invert_weights**: Boolean to invert weights (1 - weight) (default: False)
    truncate_prefix_ratio
        Ratio of words to truncate from the beginning of ground_truth (0.0-1.0).
        This prevents reward hacking by not giving credit for text the model was
        already provided as an assistant prefix. Set to 0.25 to truncate the first
        25% of words from ground_truth to match a 0.25 prefix ratio during generation.
        Default: 0.0 (no truncation).
    budget_forcing
        Budget forcing mode to truncate candidate solutions to match ground truth
        token count. Options: None (disabled, default), "tokenizer" (Qwen2.5-Math),
        "whitespace" (simple word splitting). When enabled, only evaluates/rewards
        tokens within the ground truth length, preventing reward hacking for
        generating beyond the expected length. Works in sync with truncate_prefix_ratio.
    num_workers
        Number of parallel workers for batch processing. If None, defaults to 
        value from extra_info['num_workers'] or DEFAULT_NUM_WORKERS (48).
        Set to 1 to use sequential processing (recommended with Ray to avoid deadlocks).
        Higher values enable parallel processing but may cause ProcessPoolExecutor issues.

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
    
    # Validation checks for MIA features
    if profile.get("use_mia_adaptive_matching", False) and profile.get("use_mia_weighting", False):
        warnings.warn(
            "Both use_mia_adaptive_matching and use_mia_weighting are enabled. "
            "These features are mutually exclusive. use_mia_adaptive_matching will take precedence.",
            RuntimeWarning
        )
    
    # Check if adaptive matching has multiple ground truths (only in single mode, batch mode is checked per sample)
    if profile.get("use_mia_adaptive_matching", False) and ground_truth is not None:
        # Check if ground_truth is a single string or if target_gt in extra_info is a single string
        gt_is_single = isinstance(ground_truth, str)
        target_gt = config.get("target_gt") if config else None
        target_gt_is_single = isinstance(target_gt, str)
        
        if gt_is_single and (target_gt is None or target_gt_is_single):
            warnings.warn(
                "use_mia_adaptive_matching is enabled but only a single ground truth is provided. "
                "Adaptive matching works best with multiple ground truths (e.g., from augment_target_gt). "
                "Consider providing a list of ground truths or using target_gt in extra_info.",
                RuntimeWarning
            )
    
    # Get num_workers and show_progress from config
    # Priority: function parameter > config dict > default
    if num_workers is None:
        num_workers = int(config.get("num_workers", DEFAULT_NUM_WORKERS))
    show_progress = bool(config.get("show_progress", True))
    
    # Extract truncate_prefix_ratio: prioritize function parameter, then fall back to config
    # This supports both direct passing via reward_kwargs and config-based passing
    if truncate_prefix_ratio == 0.0:
        truncate_prefix_ratio = float(config.get("truncate_prefix_ratio", 0.0))
    
    # Apply prefix truncation to ground_truth if configured
    if truncate_prefix_ratio > 0.0:
        # Handle single mode
        if ground_truth is not None and isinstance(ground_truth, str):
            ground_truth = _truncate_prefix_from_ground_truth(ground_truth, truncate_prefix_ratio)
        elif ground_truth is not None and isinstance(ground_truth, list):
            ground_truth = [_truncate_prefix_from_ground_truth(gt, truncate_prefix_ratio) for gt in ground_truth]
        
        # Handle batch mode - parallelize for large batches
        if ground_truths is not None:
            # Parallelize if batch is large enough (threshold: 2x workers)
            if len(ground_truths) >= num_workers * 2:
                from functools import partial
                ExecutorClass = _get_executor_class()
                # Use partial to bind truncate_prefix_ratio (makes function picklable for ProcessPool)
                truncate_fn = partial(_truncate_ground_truth_item, truncate_prefix_ratio=truncate_prefix_ratio)
                with ExecutorClass(max_workers=num_workers) as executor:
                    ground_truths = list(executor.map(truncate_fn, ground_truths))
            else:
                # Sequential for small batches
                ground_truths = [_truncate_ground_truth_item(gt, truncate_prefix_ratio) for gt in ground_truths]
    
    # Extract budget forcing mode: prioritize function parameter, then fall back to config
    # This supports both direct passing via reward_kwargs and config-based passing
    if budget_forcing is None:
        budget_forcing_mode = config.get("budget_forcing")
    else:
        budget_forcing_mode = budget_forcing

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
            
            # Apply budget forcing if configured
            if budget_forcing_mode and all_candidates:
                truncated_candidates = []
                for cand, ref in zip(all_candidates, all_references):
                    # Count tokens in reference (after prefix truncation)
                    if budget_forcing_mode == "tokenizer":
                        tokenizer = _get_default_tokenizer()
                        if tokenizer is not None:
                            ref_budget = len(tokenizer.tokenize(ref))
                        else:
                            ref_budget = len(ref.split())
                    else:
                        ref_budget = len(ref.split())
                    
                    # Truncate candidate to match reference budget
                    truncated_candidates.append(_truncate_to_budget(cand, ref_budget, budget_forcing_mode))
                
                all_candidates = truncated_candidates
            
            # Compute all scores in parallel
            if all_candidates:
                all_scores = _compute_scores_parallel(
                    all_references, all_candidates, profile, num_workers,
                    show_progress=show_progress,
                    desc=f"Lexical rewards ({len(sols)} samples, {len(all_candidates)} pairs)"
                )
                
                # Apply MIA adaptive matching if configured
                if profile.get("use_mia_adaptive_matching", False):
                    mia_invert = profile.get("mia_invert_weights", True)  # Default True for adaptive matching
                    mia_adaptive_mode = profile.get("mia_adaptive_mode", "linear")  # Default linear
                    mia_adaptive_variant = profile.get("mia_adaptive_variant", "standard")  # Default standard
                    
                    # Group scores by solution and apply adaptive matching
                    results = []
                    for sol_idx in range(len(sols)):
                        sol_scores = [
                            all_scores[pair_idx] 
                            for s_idx, pair_idx in pair_indices 
                            if s_idx == sol_idx and pair_idx >= 0
                        ]
                        
                        if not sol_scores:
                            results.append(0.0)
                            continue
                        
                        # Get extra_info for this solution
                        ei = defaults[sol_idx] if sol_idx < len(defaults) else None
                        
                        # Extract MIA weight (using first reference for consistency)
                        first_ref_for_sol = None
                        for s_idx, pair_idx in pair_indices:
                            if s_idx == sol_idx and pair_idx >= 0:
                                first_ref_for_sol = pair_refs[pair_idx]
                                break
                        
                        mia_weight = _extract_mia_weight(first_ref_for_sol, ei) if first_ref_for_sol else None
                        
                        if mia_weight is not None:
                            # Apply adaptive matching with variant support
                            adaptive_score = _apply_mia_adaptive_matching(
                                sol_scores, mia_weight, mia_invert, mia_adaptive_mode, mia_adaptive_variant
                            )
                            results.append(adaptive_score)
                        else:
                            # No MIA weight available, fall back to max
                            results.append(max(sol_scores))
                
                # Apply MIA weighting if configured (mutually exclusive with adaptive matching)
                elif profile.get("use_mia_weighting", False):
                    mia_mode = profile.get("mia_weighting_mode", "linear")
                    mia_invert = profile.get("mia_invert_weights", False)
                    contrastive_alpha = profile.get("mia_contrastive_alpha", 0.5)
                    
                    for i in range(len(all_scores)):
                        # Get the extra_info for this sample
                        sol_idx = pair_indices[i][0]
                        ei = defaults[sol_idx] if sol_idx < len(defaults) else None
                        
                        # Extract and apply MIA weight
                        mia_weight = _extract_mia_weight(pair_refs[i], ei)
                        if mia_weight is not None:
                            all_scores[i] = _apply_mia_weighting(
                                all_scores[i], mia_weight, mia_mode, mia_invert, contrastive_alpha
                            )
                    
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
                
                else:
                    # No MIA processing, just take maximum
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
        
        # Apply budget forcing if configured
        if budget_forcing_mode and all_candidates:
            truncated_candidates = []
            for cand, ref in zip(all_candidates, all_references):
                # Count tokens in reference (after prefix truncation)
                if budget_forcing_mode == "tokenizer":
                    tokenizer = _get_default_tokenizer()
                    if tokenizer is not None:
                        ref_budget = len(tokenizer.tokenize(ref))
                    else:
                        ref_budget = len(ref.split())
                else:
                    ref_budget = len(ref.split())
                
                # Truncate candidate to match reference budget
                truncated_candidates.append(_truncate_to_budget(cand, ref_budget, budget_forcing_mode))
            
            all_candidates = truncated_candidates
        
        # Batch process all filtered pairs in parallel
        if all_candidates:
            all_scores = _compute_scores_parallel(
                all_references, all_candidates, profile, num_workers,
                show_progress=show_progress,
                desc=f"Lexical rewards (filtered, {len(sols)} samples, {len(all_candidates)} pairs)"
            )
            
            # Apply MIA adaptive matching if configured
            if profile.get("use_mia_adaptive_matching", False):
                mia_invert = profile.get("mia_invert_weights", True)  # Default True for adaptive matching
                mia_adaptive_mode = profile.get("mia_adaptive_mode", "linear")  # Default linear
                mia_adaptive_variant = profile.get("mia_adaptive_variant", "standard")  # Default standard
                
                # Group scores by solution and apply adaptive matching
                results = []
                for sol_idx in range(len(sols)):
                    if sol_idx in empty_solutions:
                        results.append(0.0)
                        continue
                    
                    sol_scores = [
                        all_scores[i] 
                        for i, pi in enumerate(pair_indices) 
                        if pi == sol_idx
                    ]
                    
                    if not sol_scores:
                        results.append(0.0)
                        continue
                    
                    # Get extra_info for this solution
                    ei = defaults[sol_idx] if sol_idx < len(defaults) else None
                    
                    # Extract MIA weight (using first reference for consistency)
                    first_ref_for_sol = None
                    for i, pi in enumerate(pair_indices):
                        if pi == sol_idx:
                            first_ref_for_sol = pair_refs[i]
                            break
                    
                    mia_weight = _extract_mia_weight(first_ref_for_sol, ei) if first_ref_for_sol else None
                    
                    if mia_weight is not None:
                        # Apply adaptive matching with variant support
                        adaptive_score = _apply_mia_adaptive_matching(
                            sol_scores, mia_weight, mia_invert, mia_adaptive_mode, mia_adaptive_variant
                        )
                        results.append(adaptive_score)
                    else:
                        # No MIA weight available, fall back to max
                        results.append(max(sol_scores))
            
            # Apply MIA weighting if configured (mutually exclusive with adaptive matching)
            elif profile.get("use_mia_weighting", False):
                mia_mode = profile.get("mia_weighting_mode", "linear")
                mia_invert = profile.get("mia_invert_weights", False)
                contrastive_alpha = profile.get("mia_contrastive_alpha", 0.5)
                
                for i in range(len(all_scores)):
                    # Get the extra_info for this sample
                    sol_idx = pair_indices[i]
                    ei = defaults[sol_idx] if sol_idx < len(defaults) else None
                    
                    # Extract and apply MIA weight
                    mia_weight = _extract_mia_weight(pair_refs[i], ei)
                    if mia_weight is not None:
                        all_scores[i] = _apply_mia_weighting(
                            all_scores[i], mia_weight, mia_mode, mia_invert, contrastive_alpha
                        )
                
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
                # No MIA processing, just take maximum
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

    # Compute scores for all references
    metrics_to_compute = set(profile["metrics"])
    all_ref_scores = []
    
    for ref in refs:
        # Apply budget forcing if configured
        current_solution = solution_str
        if budget_forcing_mode:
            # Count tokens in reference (after prefix truncation)
            if budget_forcing_mode == "tokenizer":
                tokenizer = _get_default_tokenizer()
                if tokenizer is not None:
                    ref_budget = len(tokenizer.tokenize(ref))
                else:
                    ref_budget = len(ref.split())
            else:
                ref_budget = len(ref.split())
            
            # Truncate solution to match reference budget
            current_solution = _truncate_to_budget(solution_str, ref_budget, budget_forcing_mode)
        
        metrics = _compute_lexical_metrics(ref, current_solution, metrics_to_compute)
        score = _aggregate_metrics(metrics, profile, ref, current_solution)
        all_ref_scores.append(score)
    
    # Apply MIA adaptive matching if configured
    if profile.get("use_mia_adaptive_matching", False):
        mia_invert = profile.get("mia_invert_weights", True)  # Default True for adaptive matching
        mia_adaptive_mode = profile.get("mia_adaptive_mode", "linear")  # Default linear
        mia_adaptive_variant = profile.get("mia_adaptive_variant", "standard")  # Default standard
        
        # Extract MIA weight (using first reference for consistency)
        first_ref = refs[0] if refs else None
        mia_weight = _extract_mia_weight(first_ref, extra_info) if first_ref else None
        
        if mia_weight is not None:
            # Apply adaptive matching with variant support
            return _apply_mia_adaptive_matching(
                all_ref_scores, mia_weight, mia_invert, mia_adaptive_mode, mia_adaptive_variant
            )
        else:
            # No MIA weight available, fall back to max
            return max(all_ref_scores) if all_ref_scores else 0.0
    
    # Apply MIA weighting if configured (mutually exclusive with adaptive matching)
    elif profile.get("use_mia_weighting", False):
        best_score = max(all_ref_scores) if all_ref_scores else 0.0
        best_ref = refs[all_ref_scores.index(best_score)] if all_ref_scores else None
        
        if best_ref is not None:
            mia_weight = _extract_mia_weight(best_ref, extra_info)
            if mia_weight is not None:
                mia_mode = profile.get("mia_weighting_mode", "linear")
                mia_invert = profile.get("mia_invert_weights", False)
                contrastive_alpha = profile.get("mia_contrastive_alpha", 0.5)
                best_score = _apply_mia_weighting(
                    best_score, mia_weight, mia_mode, mia_invert, contrastive_alpha
                )
        
        return best_score
    
    # No MIA processing, just take maximum
    else:
        return max(all_ref_scores) if all_ref_scores else 0.0


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