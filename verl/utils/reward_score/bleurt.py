"""
BLEURT-based reward functions for VERL.

This module provides an easy way to compute a dense reward based on the
BLEURT score between the model response (``solution_str``) and the
reference answer (``ground_truth``). BLEURT is a learned evaluation metric
that uses BERT to assess text quality based on fluency, meaning, and
grammatical correctness.

Highlights
----------
* **BLEURT** evaluation using the bleurt-pytorch implementation
* **Length penalties** with configurable types (none, ratio, sqrt, log)
* **Configurable threshold** for when to apply length penalties  
* **Word-based** length calculation for accurate penalty computation
* **Batched** helper mirrors the API of :pyfunc:`verl.utils.reward_score.lexical.compute_score`
* **Reference filtering** support like other reward modules

Usage example
~~~~~~~~~~~~~
>>> from verl.utils.reward_score.bleurt import compute_score
>>> compute_score(
...     data_source="dummy",
...     solution_str="Cats are lovely companions.",
...     ground_truth=["Cats make great pets.", "Dogs are loyal."],
...     extra_info={"length_penalty": "ratio", "length_threshold": 1.5}
... )
0.8...

Configuration
~~~~~~~~~~~~~
You can configure BLEURT behavior via the ``extra_info`` parameter:

* ``length_penalty``: Type of penalty ("none", "ratio", "sqrt", "log")
* ``length_threshold``: Apply penalty when output exceeds reference by this factor
* ``bleurt_checkpoint``: BLEURT model checkpoint to use
* ``device``: Device to run BLEURT on ("cuda", "cpu", or None for auto)
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import List, Iterable, Tuple, Optional
import logging

import torch
import transformers

try:
    from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    _HAS_BLEURT = True
except ImportError:
    _HAS_BLEURT = False
    warnings.warn(
        "bleurt-pytorch is not installed – BLEURT reward will fall back "
        "to lexical ratio. Install it via `pip install bleurt-pytorch`.",
        RuntimeWarning,
    )

from difflib import SequenceMatcher
import re


__all__ = ["compute_score", "compute_score_batched"]

# Default configuration
_DEFAULT_CHECKPOINT = "lucadiliello/BLEURT-20"
_DEFAULT_LENGTH_PENALTY = "none"
_DEFAULT_LENGTH_THRESHOLD = 1.5

# Global model cache to avoid reloading
_MODEL_CACHE = {}


def _get_bleurt_model(checkpoint: str = _DEFAULT_CHECKPOINT, device: Optional[str] = None):
    """Get or create a BLEURT model instance with caching."""
    if not _HAS_BLEURT:
        if not hasattr(_get_bleurt_model, "_warned_no_bleurt"):
            logging.warning(
                "bleurt-pytorch is not installed - using lexical similarity fallback. "
                "Install via `pip install bleurt-pytorch` to use BLEURT scoring."
            )
            _get_bleurt_model._warned_no_bleurt = True
        return None
        
    cache_key = (checkpoint, device)
    if cache_key not in _MODEL_CACHE:
        try:
            actual_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            
            config = BleurtConfig.from_pretrained(checkpoint)
            model = BleurtForSequenceClassification.from_pretrained(checkpoint).to(actual_device)
            tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
            model.eval()
            
            _MODEL_CACHE[cache_key] = {
                'model': model,
                'tokenizer': tokenizer,
                'device': actual_device
            }
            logging.info(f"Successfully loaded BLEURT model {checkpoint} on {actual_device}")
        except Exception as e:
            if not hasattr(_get_bleurt_model, f"_warned_{checkpoint}"):
                logging.warning(
                    f"Failed to load BLEURT model {checkpoint} - using lexical similarity fallback. "
                    f"Error: {e}"
                )
                _get_bleurt_model.__dict__[f"_warned_{checkpoint}"] = True
            _MODEL_CACHE[cache_key] = None
            
    return _MODEL_CACHE[cache_key]


def _compute_length_penalty(reference: str, candidate: str, 
                          penalty_type: str = "none", 
                          threshold: float = 1.5) -> float:
    """
    Compute length penalty factor based on reference and candidate lengths.
    Returns a value between 0 and 1, where 1 means no penalty.
    
    The penalty is applied if the output length falls outside the range
    [1/threshold, threshold] * reference_length. This creates a "sweet spot"
    where outputs of similar length to the reference are not penalized.
    
    Available penalty types:
    - none: No penalty (always returns 1.0)
    - ratio: Linear penalty based on length ratio
    - sqrt: Square root of ratio for milder penalty
    - log: Logarithmic penalty based on length ratio
    - quadratic: Quadratic penalty (ratio^2) for stronger penalization
    - exponential: Exponential penalty (e^(-ratio)) for aggressive penalization
    """
    ref_len = len(reference.split())
    out_len = len(candidate.split())
    
    # Calculate both ratios to handle both longer and shorter outputs
    ratio_short = out_len / ref_len  # < 1/threshold means too short
    ratio_long = ref_len / out_len   # < 1/threshold means too long
    
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
        import math
        return min(1.0, math.log(1 + min(ref_len, out_len)) / math.log(1 + max(ref_len, out_len)))
    elif penalty_type == "quadratic":
        return min(1.0, ratio ** 2)  # Stronger penalty using square of ratio
    elif penalty_type == "exponential":
        import math
        # Exponential decay: e^(-x) where x is (1 - ratio)
        # This creates a very sharp dropoff outside the acceptable range
        return min(1.0, math.exp(-(1 - ratio)))
    else:
        warnings.warn(f"Unknown length penalty type: {penalty_type}, using 'none'")
        return 1.0


def _lexical_ratio(a: str, b: str) -> float:
    """Fallback lexical similarity when BLEURT is unavailable."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _single_bleurt_score(solution: str, reference: str, 
                        checkpoint: str = _DEFAULT_CHECKPOINT,
                        device: Optional[str] = None,
                        length_penalty: str = _DEFAULT_LENGTH_PENALTY,
                        length_threshold: float = _DEFAULT_LENGTH_THRESHOLD) -> float:
    """Compute BLEURT score between solution and reference."""
    model_info = _get_bleurt_model(checkpoint, device)
    
    # Fallback to lexical similarity if BLEURT unavailable
    if model_info is None:
        return _lexical_ratio(solution, reference)
    
    try:
        # Tokenize inputs with truncation to handle long sequences
        # BLEURT models typically have a max length of 512 tokens
        # Temporarily suppress truncation warnings
        old_level = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        
        inputs = model_info['tokenizer'](
            [reference], [solution], 
            padding='longest', 
            truncation='longest_first',  # Use longest_first strategy
            max_length=512,
            return_tensors='pt'
        )
        # Restore original logging level
        transformers.logging.set_verbosity(old_level)
        
        # Move inputs to correct device
        inputs = {k: v.to(model_info['device']) for k, v in inputs.items()}
        
        # Compute base score
        with torch.no_grad():
            outputs = model_info['model'](**inputs)
            base_score = outputs.logits.flatten().cpu().item()
        
        # Apply length penalty
        penalty = _compute_length_penalty(reference, solution, length_penalty, length_threshold)
        final_score = base_score * penalty
        
        # Normalize to [0, 1] range (BLEURT typically ranges from -2 to 2)
        # Using sigmoid-like transformation
        normalized_score = 1.0 / (1.0 + torch.exp(-torch.tensor(final_score)).item())
        
        return normalized_score
        
    except Exception as e:
        if not hasattr(_single_bleurt_score, "_warned_compute"):
            logging.warning(
                f"BLEURT computation failed - using lexical similarity fallback. Error: {e}"
            )
            _single_bleurt_score._warned_compute = True
        return _lexical_ratio(solution, reference)


def _best_bleurt_score(solution: str, references: Iterable[str], **kwargs) -> float:
    """Compute best BLEURT score against multiple references."""
    scores = [_single_bleurt_score(solution, ref, **kwargs) for ref in references]
    return max(scores, default=0.0)


# ---------------------------------------------------------------------------
# Reference filtering helper (same semantics as lexical/embedding variants)
# ---------------------------------------------------------------------------

def _filter_refs(refs: List[str], extra_info: dict | None) -> List[str]:
    """Filter references based on extra_info criteria."""
    if not extra_info or not isinstance(extra_info, dict):
        return refs
        
    # 1. Exact target string(s)
    tgt = extra_info.get("target_gt")
    if isinstance(tgt, str):
        subset = [r for r in refs if r == tgt]
        if subset:
            return subset
    elif isinstance(tgt, list):
        # Handle list of target strings - keep references that match any of them
        subset = [r for r in refs if r in tgt]
        if subset:
            return subset

    # 2. Last prompt token heuristic  
    if extra_info.get("filter_gt_by_prompt_token") and "prompt" in extra_info:
        prompt_txt = str(extra_info["prompt"]).strip()
        if prompt_txt:
            # Simple tokenization for filtering
            last_tok = prompt_txt.split()[-1].lower()
            token_re = re.compile(r"[A-Za-z0-9]+")
            subset = []
            for r in refs:
                ref_tokens = [t.lower() for t in token_re.findall(r)]
                if last_tok in ref_tokens:
                    subset.append(r)
            if subset:
                return subset

    return refs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_score(  # noqa: PLR0913
    data_source: str | List[str] | None = None,
    solution_str: str | List[str] | None = None,
    ground_truth: str | List[str] | None = None,
    extra_info: dict | List[dict] | None = None,
    *,
    data_sources: List[str] | None = None,
    solution_strs: List[str] | None = None,
    ground_truths: List[str | List[str]] | None = None,
    extra_infos: List[dict | None] | None = None,
) -> float | List[float]:
    """BLEURT-based reward (single or batched).

    Behaviour mirrors :pyfunc:`verl.utils.reward_score.lexical.compute_score`:
    • Single-sample mode: `solution_str` vs every string in `ground_truth` list → best score.
    • Batch mode: each solution in `solution_strs` is compared to the references.
    Returns a float or list[float] in the range [0, 1].
    
    Parameters
    ----------
    data_source : str | List[str] | None
        Data source identifier (ignored, kept for API compatibility)
    solution_str : str | List[str] | None  
        Model-generated solution(s)
    ground_truth : str | List[str] | None
        Reference answer(s)
    extra_info : dict | List[dict] | None
        Configuration dictionary that can contain:
        - length_penalty: "none", "ratio", "sqrt", "log" (default: "none")
        - length_threshold: float (default: 1.5)
        - bleurt_checkpoint: str (default: "lucadiliello/BLEURT-20")
        - device: str (default: auto-detect)
        - target_gt: str (for reference filtering)
        - filter_gt_by_prompt_token: bool (for reference filtering)
        - prompt: str (used with filter_gt_by_prompt_token)
    
    Returns
    -------
    float | List[float]
        BLEURT score(s) in range [0, 1]. If BLEURT is unavailable or fails,
        falls back to lexical similarity and logs a warning.
    """

    # Extract configuration from extra_info
    config = extra_info if isinstance(extra_info, dict) else {}
    length_penalty = config.get("length_penalty", _DEFAULT_LENGTH_PENALTY)
    length_threshold = config.get("length_threshold", _DEFAULT_LENGTH_THRESHOLD)
    checkpoint = config.get("bleurt_checkpoint", _DEFAULT_CHECKPOINT)
    device = config.get("device", None)
    
    bleurt_kwargs = {
        "checkpoint": checkpoint,
        "device": device,
        "length_penalty": length_penalty,
        "length_threshold": length_threshold
    }

    # Batch mode detection
    if solution_strs is not None or ground_truths is not None:
        needs_filter = False
        if extra_infos is not None:
            for ei in extra_infos:
                if isinstance(ei, dict) and (
                    "target_gt" in ei or ei.get("filter_gt_by_prompt_token")
                ):
                    needs_filter = True
                    break

        sols = solution_strs or []
        gts_flat: List[str] = []
        for gt in ground_truths or []:
            if isinstance(gt, list):
                gts_flat.extend(gt)
            else:
                gts_flat.append(gt)
        if not gts_flat:
            gts_flat.append("")

        if not needs_filter:
            return [_best_bleurt_score(s, gts_flat, **bleurt_kwargs) for s in sols]

        # per-sample path with filtering
        res: List[float] = []
        defaults = [None] * len(sols) if extra_infos is None else extra_infos
        for sol, gt, ei in zip(sols, ground_truths, defaults):
            refs = [gt] if isinstance(gt, str) else list(gt)
            refs = _filter_refs(refs, ei)
            
            # Update config for this sample
            sample_config = ei if isinstance(ei, dict) else {}
            sample_kwargs = bleurt_kwargs.copy()
            sample_kwargs.update({
                "length_penalty": sample_config.get("length_penalty", length_penalty),
                "length_threshold": sample_config.get("length_threshold", length_threshold),
                "checkpoint": sample_config.get("bleurt_checkpoint", checkpoint),
                "device": sample_config.get("device", device)
            })
            
            res.append(_best_bleurt_score(sol, refs, **sample_kwargs))
        return res

    # ---------------- Single sample path ----------------

    if solution_str is None or ground_truth is None:
        return 0.0

    refs = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    refs = _filter_refs(refs, extra_info)
    return _best_bleurt_score(str(solution_str), refs, **bleurt_kwargs)


# Convenience wrapper mimicking lexical/embedding API
def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str | List[str]],
    extra_infos: List[dict | None] | None = None,
):
    """Batched BLEURT evaluation."""
    return compute_score(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )


"""Usage Guide
-------------
1. Installation
~~~~~~~~~~~~~~~~
::

    pip install bleurt-pytorch torch

2. Configuration in VeRL YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Either let DDRL/VeRL auto-route by naming your dataset source *bleurt_match*::

    data_source: bleurt_match_custom

or specify it explicitly via ``custom_reward_function``::

    custom_reward_function:
      path: verl/utils/reward_score/bleurt.py
      name: compute_score

3. Selecting the metric at runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add to each example's *extra_info* field::

    {
        "metric": "bleurt",
        "length_penalty": "ratio",
        "length_threshold": 1.5,
        "bleurt_checkpoint": "lucadiliello/BLEURT-20"
    }

4. Length Penalty Options
~~~~~~~~~~~~~~~~~~~~~~~~~
* **none**: No length penalty applied
* **ratio**: Simple length ratio (ref_len/output_len, capped at 1)
* **sqrt**: Square root of ratio (sqrt(ref_len/output_len), capped at 1)  
* **log**: Logarithmic penalty (log(1 + ref_len)/log(1 + output_len), capped at 1)

5. Extending / replacing the backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to use a different BLEURT checkpoint:

* Set ``bleurt_checkpoint`` in extra_info to your preferred model
* Ensure the checkpoint is compatible with bleurt-pytorch
* Everything else (normalization, batching) remains unchanged
"""
