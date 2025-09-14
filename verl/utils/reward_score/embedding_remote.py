"""
Remote embedding-based reward functions for VERL.

This module provides an easy way to compute a dense reward based on the
semantic similarity between the model response (``solution_str``) and the
reference answer (``ground_truth``) using a remote embedding server.

The module connects to a Text Embeddings Inference (TEI) server hosting
models like Qwen3-Embedding-8B, allowing GPU-accelerated embeddings without
requiring GPU allocation on the reward worker.

Highlights
----------
* **Remote TEI Server** integration for GPU-accelerated embeddings
* **Qwen3-Embedding-8B** support with 4096-dimensional embeddings
* **Connection pooling** and retry logic for reliability
* **Fallback** to lexical similarity when server is unavailable
* **Batched** processing for efficient embedding computation
* **Compatible** with embedding.py API for drop-in replacement

Usage example
~~~~~~~~~~~~~
>>> from verl.utils.reward_score.embedding_remote import compute_score
>>> # Ensure EMBEDDING_SERVER_URL is set or pass server_url in extra_info
>>> compute_score(
...     data_source="dummy",
...     solution_str="Cats are lovely companions.",
...     ground_truth=["Cats make great pets.", "Dogs are loyal."],
...     extra_info={"server_url": "http://tei-server:8080"}
... )
0.8...

Environment Variables
~~~~~~~~~~~~~~~~~~~~
* ``EMBEDDING_SERVER_URL``: URL of the TEI server (e.g., http://localhost:8080)
* ``EMBEDDING_SERVER_API_KEY``: Optional API key for authentication
* ``EMBEDDING_SERVER_TIMEOUT``: Request timeout in seconds (default: 30)

To use this module, deploy a TEI server with Qwen3-Embedding-8B:
``docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest --model-id Qwen/Qwen3-Embedding-8B``
"""

from __future__ import annotations

import os
import warnings
import logging
from typing import List, Iterable, Tuple, Optional, Union, Any
from functools import lru_cache

import numpy as np

# Try to import embedding client
try:
    from .embedding_client import EmbeddingClient, get_default_client
    _HAS_CLIENT = True
except ImportError:
    # Try absolute import as fallback
    try:
        from embedding_client import EmbeddingClient, get_default_client
        _HAS_CLIENT = True
    except ImportError:
        _HAS_CLIENT = False
        warnings.warn(
            "embedding_client not available. Remote embeddings will fall back to lexical similarity.",
            RuntimeWarning
        )

# Import fallback lexical similarity
from difflib import SequenceMatcher
import re

logger = logging.getLogger(__name__)

__all__ = ["compute_score", "compute_score_batched"]

# Default configuration for length penalty (same as embedding.py)
_DEFAULT_LENGTH_PENALTY = "none"
_DEFAULT_LENGTH_THRESHOLD = 1.5

# Token regex (same as embedding.py)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Global client instance
_GLOBAL_CLIENT: Optional[Any] = None


def _get_client(server_url: Optional[str] = None, **kwargs) -> Optional[Any]:
    """Get or create an embedding client."""
    global _GLOBAL_CLIENT
    
    if not _HAS_CLIENT:
        return None
    
    if server_url:
        # Create a new client for specific server
        try:
            return EmbeddingClient(server_url=server_url, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create embedding client for {server_url}: {e}")
            return None
    
    # Use global client
    if _GLOBAL_CLIENT is None:
        try:
            _GLOBAL_CLIENT = get_default_client()
        except Exception as e:
            logger.error(f"Failed to get default embedding client: {e}")
            return None
    
    return _GLOBAL_CLIENT


def _tokenise(text: str) -> List[str]:
    """Tokenize text (same as embedding.py)."""
    return _TOKEN_RE.findall(text.lower())


def _lexical_ratio(a: str, b: str) -> float:
    """Fallback lexical similarity when remote embedding unavailable."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _sent_emb_remote(
    sentence: str,
    client: Any,
    cache: Optional[dict] = None
) -> Optional[np.ndarray]:
    """Get sentence embedding from remote server."""
    if client is None:
        return None
    
    # Check cache first
    if cache is not None and sentence in cache:
        return cache[sentence]
    
    try:
        # Get embedding from server
        embeddings = client.embed_texts([sentence], normalize=True, truncate=True)
        if embeddings is None or len(embeddings) == 0:
            return None
        
        embedding = embeddings[0].astype(np.float32)
        
        # Cache the result
        if cache is not None:
            cache[sentence] = embedding
        
        return embedding
        
    except Exception as e:
        logger.warning(f"Error getting remote embedding: {e}")
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity (same as embedding.py)."""
    if not a.any() or not b.any():
        return 0.0
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return 0.0
    return (num / den + 1.0) / 2.0  # map [-1,1] → [0,1]


def _compute_length_penalty(
    reference: str, candidate: str,
    penalty_type: str = "none",
    threshold: float = 1.5
) -> float:
    """
    Compute length penalty factor (same as embedding.py).
    Returns a value between 0 and 1, where 1 means no penalty.
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
        return min(1.0, ratio ** 2)
    elif penalty_type == "exponential":
        import math
        return min(1.0, math.exp(-(1 - ratio)))
    else:
        warnings.warn(f"Unknown length penalty type: {penalty_type}, using 'none'")
        return 1.0


def _filter_refs(refs: List[str], extra_info: dict | None) -> List[str]:
    """Filter references (same as embedding.py)."""
    if not extra_info or not isinstance(extra_info, dict):
        return refs
    
    # 1. Exact target string(s)
    tgt = extra_info.get("target_gt")
    if isinstance(tgt, str):
        subset = [r for r in refs if r == tgt]
        if subset:
            return subset
    elif isinstance(tgt, list):
        subset = [r for r in refs if r in tgt]
        if subset:
            return subset
    
    # 2. Last prompt token heuristic
    if extra_info.get("filter_gt_by_prompt_token") and "prompt" in extra_info:
        prompt_txt = str(extra_info["prompt"]).strip()
        if prompt_txt:
            last_tok = prompt_txt.split()[-1].lower()
            subset = [r for r in refs if last_tok in _tokenise(r)]
            if subset:
                return subset
    
    return refs


def _compute_remote_similarity_batch(
    solutions: List[str],
    references: List[str],
    client: Any,
    length_penalty: str = "none",
    length_threshold: float = 1.5
) -> List[float]:
    """
    Compute similarities for multiple solution-reference pairs efficiently.
    
    This batches all texts to the embedding server in one request.
    """
    if client is None:
        # Fallback to lexical similarity
        return [_lexical_ratio(sol, ref) for sol, ref in zip(solutions, references)]
    
    # Collect all unique texts to embed
    all_texts = list(set(solutions + references))
    
    # Get embeddings in one batch
    try:
        embeddings = client.embed_texts(all_texts, normalize=True, truncate=True)
        if embeddings is None:
            # Fallback to lexical
            return [_lexical_ratio(sol, ref) for sol, ref in zip(solutions, references)]
        
        # Create embedding lookup
        embedding_dict = {text: emb for text, emb in zip(all_texts, embeddings)}
        
        # Compute similarities
        similarities = []
        for sol, ref in zip(solutions, references):
            sol_emb = embedding_dict.get(sol)
            ref_emb = embedding_dict.get(ref)
            
            if sol_emb is None or ref_emb is None:
                similarities.append(_lexical_ratio(sol, ref))
            else:
                base_score = _cosine(sol_emb, ref_emb)
                penalty = _compute_length_penalty(ref, sol, length_penalty, length_threshold)
                similarities.append(base_score * penalty)
        
        return similarities
        
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        # Fallback to lexical
        return [_lexical_ratio(sol, ref) for sol, ref in zip(solutions, references)]


def _single_similarity(
    sol: str, ref: str,
    client: Any,
    cache: Optional[dict] = None,
    length_penalty: str = "none",
    length_threshold: float = 1.5
) -> float:
    """Compute similarity between two texts."""
    if client is None:
        return _lexical_ratio(sol, ref)
    
    # Try to get embeddings
    sol_emb = _sent_emb_remote(sol, client, cache)
    ref_emb = _sent_emb_remote(ref, client, cache)
    
    if sol_emb is None or ref_emb is None:
        # Fallback to lexical
        return _lexical_ratio(sol, ref)
    
    # Compute cosine similarity
    base_score = _cosine(sol_emb, ref_emb)
    
    # Apply length penalty
    penalty = _compute_length_penalty(ref, sol, length_penalty, length_threshold)
    return base_score * penalty


def _best_similarity(
    sol: str, refs: Iterable[str],
    client: Any,
    cache: Optional[dict] = None,
    length_penalty: str = "none",
    length_threshold: float = 1.5
) -> float:
    """Find best similarity among multiple references."""
    refs_list = list(refs)
    if not refs_list:
        return 0.0
    
    if client is None:
        return max((_lexical_ratio(sol, r) for r in refs_list), default=0.0)
    
    # For efficiency, batch all embeddings
    all_texts = [sol] + refs_list
    try:
        embeddings = client.embed_texts(all_texts, normalize=True, truncate=True)
        if embeddings is None or len(embeddings) != len(all_texts):
            # Fallback
            return max((_lexical_ratio(sol, r) for r in refs_list), default=0.0)
        
        sol_emb = embeddings[0]
        ref_embs = embeddings[1:]
        
        # Find best match
        best_score = 0.0
        best_ref_idx = 0
        for idx, ref_emb in enumerate(ref_embs):
            score = _cosine(sol_emb, ref_emb)
            if score > best_score:
                best_score = score
                best_ref_idx = idx
        
        # Apply length penalty to best match
        penalty = _compute_length_penalty(refs_list[best_ref_idx], sol, length_penalty, length_threshold)
        return best_score * penalty
        
    except Exception as e:
        logger.warning(f"Batch similarity computation failed: {e}")
        # Fallback to sequential computation
        return max(
            (_single_similarity(sol, r, client, cache, length_penalty, length_threshold) for r in refs_list),
            default=0.0
        )


def compute_score(
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
    """Remote embedding similarity reward (single or batched).

    Behaviour mirrors :pyfunc:`verl.utils.reward_score.embedding.compute_score`:
    • Single-sample mode: `solution_str` vs every string in `ground_truth` list → best similarity.
    • Batch mode: each solution in `solution_strs` is compared to the references.
    Returns a float or list[float] in the range [0, 1].
    
    Remote Server Configuration (via extra_info):
    - server_url: TEI server URL (overrides EMBEDDING_SERVER_URL)
    - api_key: Optional API key (overrides EMBEDDING_SERVER_API_KEY)
    - timeout: Request timeout in seconds
    
    Length penalty configuration (via extra_info):
    - length_penalty: "none", "ratio", "sqrt", "log", "quadratic", "exponential" (default: "none")
    - length_threshold: float (default: 1.5)
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
    
    # Extract server configuration
    server_url = config.get("server_url")
    api_key = config.get("api_key")
    timeout = float(config.get("timeout", os.getenv("EMBEDDING_SERVER_TIMEOUT", "30")))
    
    # Extract length penalty configuration
    length_penalty = config.get("length_penalty", _DEFAULT_LENGTH_PENALTY)
    length_threshold = config.get("length_threshold", _DEFAULT_LENGTH_THRESHOLD)
    
    # Get or create client
    client = _get_client(server_url=server_url, api_key=api_key, timeout=timeout)
    
    # Log if falling back to lexical
    if client is None:
        logger.warning("No embedding server available, falling back to lexical similarity")
    
    # Create a cache for this request
    cache = {}
    
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
        
        if not needs_filter and client is not None:
            # Optimized batch processing without filtering
            # Create all solution-reference pairs
            all_pairs = []
            for sol in sols:
                # Find best reference for each solution
                best_score = 0.0
                best_ref = gts_flat[0]
                
                # Quick optimization: if we have many references, batch process
                if len(gts_flat) > 5:
                    scores = _compute_remote_similarity_batch(
                        [sol] * len(gts_flat), gts_flat, client,
                        length_penalty, length_threshold
                    )
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    best_score = scores[best_idx]
                else:
                    # For few references, use sequential processing
                    best_score = _best_similarity(
                        sol, gts_flat, client, cache,
                        length_penalty, length_threshold
                    )
                
                all_pairs.append(best_score)
            
            return all_pairs
        
        # Filtered or fallback path
        res: List[float] = []
        defaults = [None] * len(sols) if extra_infos is None else extra_infos
        for sol, gt, ei in zip(sols, ground_truths, defaults):
            refs = [gt] if isinstance(gt, str) else list(gt)
            refs = _filter_refs(refs, ei)
            
            # Extract per-sample config
            sample_penalty = length_penalty
            sample_threshold = length_threshold
            if isinstance(ei, dict):
                sample_penalty = ei.get("length_penalty", length_penalty)
                sample_threshold = ei.get("length_threshold", length_threshold)
            
            res.append(_best_similarity(
                sol, refs, client, cache,
                sample_penalty, sample_threshold
            ))
        return res
    
    # ---------------- Single sample path ----------------
    
    if solution_str is None or ground_truth is None:
        return 0.0
    
    refs = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    refs = _filter_refs(refs, extra_info)
    
    return _best_similarity(
        str(solution_str), refs, client, cache,
        length_penalty, length_threshold
    )


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str | List[str]],
    extra_infos: List[dict | None] | None = None,
):
    """Convenience wrapper for batched remote embedding evaluation."""
    return compute_score(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )


"""Usage Guide
-------------
1. Server Deployment
~~~~~~~~~~~~~~~~~~~~
Deploy a Text Embeddings Inference (TEI) server with Qwen3-Embedding-8B:

Using Docker::

    docker run -d -p 8080:80 \\
      --gpus all \\
      --name qwen3-embedding \\
      ghcr.io/huggingface/text-embeddings-inference:latest \\
      --model-id Qwen/Qwen3-Embedding-8B \\
      --max-batch-tokens 65536

Using Kubernetes::

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: qwen3-embedding
    spec:
      replicas: 1
      template:
        spec:
          containers:
          - name: tei
            image: ghcr.io/huggingface/text-embeddings-inference:latest
            args:
              - --model-id=Qwen/Qwen3-Embedding-8B
              - --max-batch-tokens=65536
            ports:
            - containerPort: 80
            resources:
              limits:
                nvidia.com/gpu: 1

2. Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set the embedding server URL::

    export EMBEDDING_SERVER_URL="http://qwen3-embedding-service:8080"
    
    # Optional: if using authentication
    export EMBEDDING_SERVER_API_KEY="your-api-key"
    
    # Optional: custom timeout
    export EMBEDDING_SERVER_TIMEOUT="60"

3. Configuration in VeRL YAML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Either let VERL auto-route by naming your dataset source *embedding_remote*::

    data_source: embedding_remote_custom

or specify it explicitly via ``custom_reward_function``::

    custom_reward_function:
      path: verl/utils/reward_score/embedding_remote.py
      name: compute_score

4. Runtime Configuration
~~~~~~~~~~~~~~~~~~~~~~~~
Pass server configuration via extra_info::

    extra_info = {
        "server_url": "http://custom-server:8080",  # Optional, overrides env
        "api_key": "custom-key",                     # Optional
        "timeout": 45,                               # Optional
        "length_penalty": "ratio",                   # Optional
        "length_threshold": 2.0                      # Optional
    }

5. Network Connectivity Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Option A - Kubernetes Service Discovery::

    export EMBEDDING_SERVER_URL="http://qwen3-embedding-service.default.svc.cluster.local:8080"

Option B - Direct IP/Hostname::

    export EMBEDDING_SERVER_URL="http://10.0.0.5:8080"

Option C - External Load Balancer::

    export EMBEDDING_SERVER_URL="https://embeddings.example.com"

6. Monitoring
~~~~~~~~~~~~~
Check server health::

    curl http://your-server:8080/health
    
Get server info::

    curl http://your-server:8080/info

7. Graceful Degradation
~~~~~~~~~~~~~~~~~~~~~~~
If the embedding server is unavailable, the module automatically falls back
to lexical similarity (difflib.SequenceMatcher) to ensure training continues.
Monitor logs for fallback warnings.
"""
