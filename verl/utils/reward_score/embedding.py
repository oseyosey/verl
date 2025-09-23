"""
Embedding similarity–based reward functions for VERL.

This module provides an easy way to compute a dense reward based on the
semantic similarity between the model response (``solution_str``) and the
reference answer (``ground_truth``).  The default backend is **FastText** –
each sentence is represented as the average of its token embeddings and the
reward is the cosine similarity (mapped to the range [0, 1]).

Highlights
----------
* **FastText** is used by default via either ``gensim``'s downloader
  (``fasttext-wiki-news-subwords-300``) *or* the official FastText Python
  package if you point the ``FASTTEXT_MODEL`` env-var to a ``.bin`` model.
* **Qwen3-Embedding-0.6B** is available as an alternative via sentence-transformers
  when specified in the metric field as "qwen3". Uses optimized query/document
  encoding with built-in similarity computation for better performance.
* **Fallback** to a simple lexical ratio when no embedding model is
  available so that training can still proceed.
* **Batched** helper mirrors the API of :pyfunc:`verl.utils.reward_score.lexical.compute_score`.

Usage example
~~~~~~~~~~~~~
>>> from verl.utils.reward_score.embedding import compute_score
>>> compute_score(
...     data_source="dummy",
...     solution_str="Cats are lovely companions.",
...     ground_truth=["Cats make great pets.", "Dogs are loyal."],
... )
0.8...

You can set the environment variable ``FASTTEXT_MODEL`` to an absolute path
pointing to a ``.bin`` model downloaded from https://fasttext.cc/ to avoid the
`gensim` download step.

To use Qwen3-Embedding-0.6B, set the metric in extra_info:
>>> compute_score(
...     data_source="dummy",
...     solution_str="Cats are lovely companions.",
...     ground_truth=["Cats make great pets.", "Dogs are loyal."],
...     extra_info={"metric": "qwen3"}
... )
"""

from __future__ import annotations

import os
import pdb
import warnings
from functools import lru_cache
from typing import List, Iterable, Tuple, Optional, Union, Any

import numpy as np

# Global variables for different embedding models
_MODEL: Optional[Any] = None
_EMBED_DIM: int = 0
_MODEL_TYPE: str = "fasttext"  # Default model type

# Default configuration for length penalty
_DEFAULT_LENGTH_PENALTY = "none"
_DEFAULT_LENGTH_THRESHOLD = 1.5

# Try to load sentence-transformers for Qwen3 support
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load FastText model as default
try:
    # Prefer the *official* FastText wrapper when a binary model is available.
    import fasttext  # type: ignore

    _FASTTEXT_BIN: str | None = os.getenv("FASTTEXT_MODEL")
    if _FASTTEXT_BIN and os.path.isfile(_FASTTEXT_BIN):
        _MODEL = fasttext.load_model(_FASTTEXT_BIN)
        _EMBED_DIM = _MODEL.get_dimension()
    else:
        raise ModuleNotFoundError  # trigger gensim fallback
except (ModuleNotFoundError, ValueError, ImportError):
    try:
        # Fallback to gensim-downloader – small subset (~1 GB) but easy.
        import gensim.downloader as api  # type: ignore

        _MODEL = api.load("fasttext-wiki-news-subwords-300")
        _EMBED_DIM = 300
    except (ImportError, ValueError, Exception):  # noqa: BLE001
        _MODEL = None  # will use lexical fallback
        _EMBED_DIM = 0
        warnings.warn(
            "No FastText model is available – embedding reward will fall back "
            "to difflib.SequenceMatcher (lexical ratio). Install either the "
            "fasttext package and set FASTTEXT_MODEL or gensim with the "
            "'fasttext-wiki-news-subwords-300' dataset.",
            RuntimeWarning,
        )

from difflib import SequenceMatcher
import re

__all__ = ["compute_score", "compute_score_batched"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Cache for different models
_MODEL_CACHE: dict[str, Any] = {}
_EMBED_DIM_CACHE: dict[str, int] = {}


def _load_qwen3_model(model_size: str = "0.6B") -> Tuple[Any, int]:
    """Load Qwen3-Embedding model using sentence-transformers with optimizations.
    
    Parameters
    ----------
    model_size : str
        Model size to load: "0.6B", "4B", or "8B" (default: "0.6B")
    
    Returns
    -------
    Tuple[Any, int]
        The loaded model and its embedding dimension
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            f"sentence-transformers is required for Qwen3-Embedding-{model_size}. "
            "Install with: pip install sentence-transformers"
        )
    
    # Validate model size
    valid_sizes = ["0.6B", "4B", "8B"]
    if model_size not in valid_sizes:
        raise ValueError(f"Invalid model size '{model_size}'. Must be one of: {valid_sizes}")
    
    model_name = f"Qwen/Qwen3-Embedding-{model_size}"
    if model_name not in _MODEL_CACHE:
        import torch
        
        # Check if CUDA is available for FlashAttention2
        cuda_available = torch.cuda.is_available()
        
        # Debug information
        print(f"[DEBUG] Loading Qwen3 model - CUDA available: {cuda_available}")
        if cuda_available:
            print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
            print(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
        
        if cuda_available:
            # Try FlashAttention2 optimizations first (GPU only)
            try:
                print("[DEBUG] Attempting to load with FlashAttention2...")
                # Force GPU usage by setting device explicitly if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[DEBUG] Using device: {device}")
                
                model = SentenceTransformer(
                    model_name,
                    model_kwargs={
                        "attn_implementation": "flash_attention_2", 
                        # Use single device instead of auto to prevent multi-GPU distribution
                        "device_map": {"": "cuda:0"},
                        "dtype": torch.float16
                    },
                    tokenizer_kwargs={"padding_side": "left"},
                )
                _MODEL_CACHE[model_name] = model
                # Embedding dimensions for different model sizes
                embed_dims = {"0.6B": 1024, "4B": 2560, "8B": 4096}
                _EMBED_DIM_CACHE[model_name] = embed_dims[model_size]
                print(f"[DEBUG] Successfully loaded {model_name} with FlashAttention2")
                return _MODEL_CACHE[model_name], _EMBED_DIM_CACHE[model_name]
            except Exception as e:
                warnings.warn(f"Failed to load Qwen3 with FlashAttention2: {e}. Trying bfloat16...")
                try:
                    print("[DEBUG] Attempting to load with bfloat16 FlashAttention2...")
                    model = SentenceTransformer(
                        model_name,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            # Use single device instead of auto to prevent multi-GPU distribution
                        "device_map": {"": "cuda:0"},
                            "dtype": torch.bfloat16
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                    )
                    _MODEL_CACHE[model_name] = model
                    # Embedding dimensions for different model sizes
                    embed_dims = {"0.6B": 1024, "4B": 2560, "8B": 4096}
                    _EMBED_DIM_CACHE[model_name] = embed_dims[model_size]
                    print(f"[DEBUG] Successfully loaded {model_name} with bfloat16 FlashAttention2")
                    return _MODEL_CACHE[model_name], _EMBED_DIM_CACHE[model_name]
                except Exception as e2:
                    warnings.warn(f"Failed to load Qwen3 with bfloat16 FlashAttention2: {e2}. Trying basic loading...")
        
        # Fallback to basic loading (works on both CPU and GPU)
        try:
            if cuda_available:
                # On GPU, try with device_map but without FlashAttention2
                print("[DEBUG] Attempting to load on GPU without FlashAttention2...")
                # Ensure we can access GPU even without dedicated allocation
                torch.cuda.set_device(0)  # Use first available GPU
                print(f"[DEBUG] Set CUDA device to: {torch.cuda.current_device()}")
                
                model = SentenceTransformer(
                    model_name,
                    model_kwargs={
                        # Use single device instead of auto to prevent multi-GPU distribution
                        "device_map": {"": "cuda:0"},
                        "dtype": torch.float32  # Use float32 for better compatibility
                    },
                    tokenizer_kwargs={"padding_side": "left"},
                )
                print("[DEBUG] Successfully loaded on GPU without FlashAttention2")
            else:
                # On CPU, use basic loading
                print("[DEBUG] Attempting to load on CPU...")
                model = SentenceTransformer(
                    model_name,
                    tokenizer_kwargs={"padding_side": "left"},
                )
                print("[DEBUG] Successfully loaded on CPU")
            
            _MODEL_CACHE[model_name] = model
            # Embedding dimensions for different model sizes
            embed_dims = {"0.6B": 1024, "4B": 2560, "8B": 4096}
            _EMBED_DIM_CACHE[model_name] = embed_dims[model_size]
        except Exception as e3:
            raise RuntimeError(f"Failed to load {model_name}: {e3}")
    
    return _MODEL_CACHE[model_name], _EMBED_DIM_CACHE[model_name]


def _get_model_for_metric(metric: Optional[str]) -> Tuple[Any, int, str]:
    """Get the appropriate model based on the metric specified.
    
    Parameters
    ----------
    metric : Optional[str]
        The metric to use. Can be:
        - None or "fasttext": Use FastText embeddings
        - "qwen3": Use Qwen3-Embedding-0.6B (default)
        - "qwen3-0.6B": Use Qwen3-Embedding-0.6B
        - "qwen3-4B": Use Qwen3-Embedding-4B
        - "qwen3-8B": Use Qwen3-Embedding-8B
    
    Returns
    -------
    Tuple[Any, int, str]
        The model, embedding dimension, and model type
    """
    if metric and metric.startswith("qwen3"):
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Qwen3-Embedding requested but sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
        
        # Parse model size from metric
        if metric == "qwen3":
            model_size = "0.6B"  # Default to 0.6B for backward compatibility
        elif metric in ["qwen3-0.6B", "qwen3-4B", "qwen3-8B"]:
            model_size = metric.split("-")[1]
        else:
            raise ValueError(
                f"Invalid Qwen3 metric '{metric}'. Valid options are: "
                "'qwen3', 'qwen3-0.6B', 'qwen3-4B', 'qwen3-8B'"
            )
        
        # Check cache first to avoid reloading the model
        model_name = f"Qwen/Qwen3-Embedding-{model_size}"
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name], _EMBED_DIM_CACHE[model_name], "qwen3"
        
        try:
            model, embed_dim = _load_qwen3_model(model_size)
            return model, embed_dim, "qwen3"
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3-Embedding-{model_size}: {e}")
    elif metric is None or metric == "fasttext":
        # Default to FastText (None or explicit "fasttext")
        if _MODEL is None:
            raise RuntimeError(
                "No embedding model available. Install either fasttext or gensim with "
                "'fasttext-wiki-news-subwords-300' dataset, or use Qwen3 with sentence-transformers."
            )
        return _MODEL, _EMBED_DIM, "fasttext"
    else:
        # Invalid metric provided
        raise ValueError(
            f"Invalid metric '{metric}'. Supported metrics are: 'fasttext' (default), "
            f"'qwen3', 'qwen3-0.6B', 'qwen3-4B', 'qwen3-8B'. "
            f"Use None or omit the metric parameter to use the default FastText model."
        )


def _tokenise(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1024)
def _word_vec(word: str, model: Any, model_type: str, embed_dim: int) -> np.ndarray:
    """Return embedding vector for *word* or a zero-vector if OOV / model missing."""

    if model is None:
        return np.zeros(embed_dim, dtype=np.float32)
    
    if model_type == "qwen3":
        # For sentence-transformers models, we don't use word-level embeddings
        # This function is mainly for FastText compatibility
        return np.zeros(embed_dim, dtype=np.float32)
    
    # Support both the official fastText API and gensim KeyedVectors fallback
    try:
        if hasattr(model, "get_word_vector"):
            return model.get_word_vector(word)  # type: ignore[attr-defined]
        # gensim KeyedVectors expose vectors via __getitem__ or get_vector
        if hasattr(model, "get_vector"):
            return model.get_vector(word)  # type: ignore[attr-defined]
        return model[word]  # type: ignore[index]
    except (KeyError, Exception):  # noqa: BLE001
        return np.zeros(embed_dim, dtype=np.float32)


def _sent_emb(sentence: str, model: Any, model_type: str, embed_dim: int) -> np.ndarray:
    """Get sentence embedding using the specified model."""
    if model is None:
        return np.zeros(embed_dim, dtype=np.float32)
    
    if model_type == "qwen3":
        # Use sentence-transformers for Qwen3
        try:
            # sentence-transformers expects a list of sentences
            embeddings = model.encode([sentence])
            return embeddings[0].astype(np.float32)
        except Exception as e:
            warnings.warn(f"Error encoding sentence with Qwen3: {e}")
            return np.zeros(embed_dim, dtype=np.float32)
    else:
        # Use FastText approach (word-level averaging)
        tokens = _tokenise(sentence)
        if not tokens:
            return np.zeros(embed_dim, dtype=np.float32)
        vecs = [_word_vec(tok, model, model_type, embed_dim) for tok in tokens]
        return np.mean(vecs, axis=0)


def _compute_qwen3_similarity(sol: str, refs: List[str], model: Any) -> float:
    """Compute similarity using Qwen3's optimized similarity function."""
    try:
        # Use the solution as query with prompt for better performance
        query_embeddings = model.encode([sol], prompt_name="query")
        # Use references as documents
        document_embeddings = model.encode(refs)
        
        # Use the model's built-in similarity computation
        similarities = model.similarity(query_embeddings, document_embeddings)
        
        # Get the maximum similarity and convert to [0, 1] range
        # The similarity is already in [-1, 1] range, so we map it to [0, 1]
        max_sim = float(similarities[0].max())
        return (max_sim + 1.0) / 2.0  # map [-1,1] → [0,1]
        
    except Exception as e:
        raise RuntimeError(f"Error computing Qwen3 similarity: {e}")


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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() or not b.any():
        return 0.0
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return 0.0
    return (num / den + 1.0) / 2.0  # map [-1,1] → [0,1]


# ---------------------------------------------------------------------------
# Reference filtering helper (same semantics as lexical variant)
# ---------------------------------------------------------------------------


def _filter_refs(refs: List[str], extra_info: dict | None) -> List[str]:
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
            # TODO: Currently this is assuming we hardcoded. the last token of the prompt as the target token.
            # TODO: This may not be correct. 
            last_tok = prompt_txt.split()[-1].lower()
            subset = [r for r in refs if last_tok in _tokenise(r)]
            if subset:
                return subset

    return refs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _lexical_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _single_similarity(sol: str, ref: str, model: Any, model_type: str, embed_dim: int, 
                      length_penalty: str = "none", length_threshold: float = 1.5) -> float:
    if model is None:
        base_score = _lexical_ratio(sol, ref)
    elif model_type == "qwen3":
        # Use optimized Qwen3 similarity computation
        base_score = _compute_qwen3_similarity(sol, [ref], model)
    else:
        # Use FastText approach
        base_score = _cosine(_sent_emb(sol, model, model_type, embed_dim), _sent_emb(ref, model, model_type, embed_dim))
    
    # Apply length penalty
    penalty = _compute_length_penalty(ref, sol, length_penalty, length_threshold)
    return base_score * penalty


def _best_similarity(sol: str, refs: Iterable[str], model: Any, model_type: str, embed_dim: int,
                    length_penalty: str = "none", length_threshold: float = 1.5) -> float:
    if model is None:
        return max((_lexical_ratio(sol, r) for r in refs), default=0.0)
    
    if model_type == "qwen3":
        # Use optimized Qwen3 similarity computation for all references at once
        refs_list = list(refs)
        if not refs_list:
            return 0.0
        base_score = _compute_qwen3_similarity(sol, refs_list, model)
        # For Qwen3, we need to apply length penalty to the best reference
        # Find the best reference by computing individual similarities
        best_ref = max(refs_list, key=lambda r: _single_similarity(sol, r, model, model_type, embed_dim, "none", 1.5))
        penalty = _compute_length_penalty(best_ref, sol, length_penalty, length_threshold)
        return base_score * penalty
    else:
        # Use FastText approach
        return max((_single_similarity(sol, r, model, model_type, embed_dim, length_penalty, length_threshold) for r in refs), default=0.0)


# The function signature mirrors verl.utils.reward_score.lexical.compute_score

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
    """Embedding similarity reward (single or batched).

    Behaviour mirrors :pyfunc:`verl.utils.reward_score.lexical.compute_score`:
    • Single-sample mode: `solution_str` vs every string in `ground_truth` list → best similarity.
    • Batch mode: each solution in `solution_strs` is compared to the *flattened* pool of references.
    Returns a float or list[float] in the range [0, 1].
    
    Supported metrics:
    - "fasttext" (default): Uses FastText embeddings
    - "qwen3": Uses Qwen3-Embedding-0.6B via sentence-transformers (default Qwen3 size)
    - "qwen3-0.6B": Uses Qwen3-Embedding-0.6B explicitly
    - "qwen3-4B": Uses Qwen3-Embedding-4B 
    - "qwen3-8B": Uses Qwen3-Embedding-8B
    
    Length penalty configuration (via extra_info):
    - length_penalty: "none", "ratio", "sqrt", "log", "quadratic", "exponential" (default: "none")
    - length_threshold: float (default: 1.5)
    """

    # Determine the metric to use
    metric = None
    if isinstance(extra_info, dict):
        metric = extra_info.get("metric")
    elif extra_infos is not None and len(extra_infos) > 0:
        # In batched mode, get metric from first extra_info
        first_extra_info = extra_infos[0]
        if isinstance(first_extra_info, dict):
            metric = first_extra_info.get("metric")
    
    # Get the appropriate model
    model, embed_dim, model_type = _get_model_for_metric(metric)

    # Extract length penalty configuration
    length_penalty = _DEFAULT_LENGTH_PENALTY
    length_threshold = _DEFAULT_LENGTH_THRESHOLD
    
    if isinstance(extra_info, dict):
        length_penalty = extra_info.get("length_penalty", _DEFAULT_LENGTH_PENALTY)
        length_threshold = extra_info.get("length_threshold", _DEFAULT_LENGTH_THRESHOLD)
    elif extra_infos is not None and len(extra_infos) > 0:
        # In batched mode, get from first extra_info
        first_extra_info = extra_infos[0]
        if isinstance(first_extra_info, dict):
            length_penalty = first_extra_info.get("length_penalty", _DEFAULT_LENGTH_PENALTY)
            length_threshold = first_extra_info.get("length_threshold", _DEFAULT_LENGTH_THRESHOLD)

    # pdb.set_trace()
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
            return [_best_similarity(s, gts_flat, model, model_type, embed_dim, length_penalty, length_threshold) for s in sols]

        # per-sample path with filtering - use the same model for all samples
        res: List[float] = []
        defaults = [None] * len(sols) if extra_infos is None else extra_infos
        for sol, gt, ei in zip(sols, ground_truths, defaults):
            refs = [gt] if isinstance(gt, str) else list(gt)
            refs = _filter_refs(refs, ei)
            res.append(_best_similarity(sol, refs, model, model_type, embed_dim, length_penalty, length_threshold))
        return res

    # ---------------- Single sample path ----------------

    if solution_str is None or ground_truth is None:
        return 0.0

    refs = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    refs = _filter_refs(refs, extra_info)
    return _best_similarity(str(solution_str), refs, model, model_type, embed_dim, length_penalty, length_threshold)


# Convenience wrapper mimicking lexical API

def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str | List[str]],
    extra_infos: List[dict | None] | None = None,
):
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
For FastText (default)::

    pip install fasttext gensim numpy

For Qwen3-Embedding-0.6B::

    pip install sentence-transformers

(Optional) Download a full‐sized FastText model and set the environment variable::

    export FASTTEXT_MODEL=/path/to/cc.en.300.bin

2. Configuration in VeRL YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Either let DDRL/VeRL auto-route by naming your dataset source *embedding_match*::

    data_source: embedding_match_custom

or specify it explicitly via ``custom_reward_function``::

    custom_reward_function:
      path: verl/utils/reward_score/embedding.py
      name: compute_score

3. Selecting the metric at runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Add to each example's *extra_info* field::

    {"metric": "fasttext"}    # Default FastText model
    {"metric": "qwen3"}       # Qwen3-Embedding-0.6B model (default Qwen3)
    {"metric": "qwen3-0.6B"}  # Qwen3-Embedding-0.6B model (explicit)
    {"metric": "qwen3-4B"}    # Qwen3-Embedding-4B model
    {"metric": "qwen3-8B"}    # Qwen3-Embedding-8B model

4. Available Models
~~~~~~~~~~~~~~~~~~~
- **fasttext** (default): Uses FastText embeddings via gensim or fasttext package
- **qwen3** family: Uses Qwen3-Embedding models via sentence-transformers with optimizations:
  - **qwen3** or **qwen3-0.6B**: 1024-dimensional embeddings (smallest model)
  - **qwen3-4B**: 2560-dimensional embeddings (medium model) 
  - **qwen3-8B**: 4096-dimensional embeddings (largest model)
  - All models use query prompts for better performance
  - Leverage built-in similarity computation
  - Enable flash_attention_2 and optimized tokenization when available

5. Extending / replacing the backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to use a different sentence model (e.g. other Sentence-Transformers models):

* Add a new model type in *_get_model_for_metric* function
* Add corresponding loading logic in *_load_*_model* functions
* Update *_sent_emb* to handle the new model type
* Everything else (normalisation, batching) remains unchanged.
""" 