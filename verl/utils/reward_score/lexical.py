from __future__ import annotations

"""Lexical similarity–based reward functions.

This module offers a simple way to evaluate the lexical similarity between the
model response (``solution_str``) and ground-truth answer (``ground_truth``).
The main entry-point is :pyfunc:`compute_score`, which follows the interface
expected by verl's reward loading utilities (see
`verl.trainer.ppo.reward.get_custom_reward_fn`).

Key features
------------
* **Token-based metrics** are the default, focusing on exact token matching
  which is ideal for reconstruction tasks.
* **Multiple metrics** available:
  - ``token_ratio``: Jaccard similarity (default)
  - ``token_precision``, ``token_recall``, ``token_f1``: Standard IR metrics
  - ``ordered_token``: Considers token ordering via LCS
  - ``bm25``: Traditional BM25 (kept for reference)
  - ``ratio``: Character-based similarity
* **Extensible** – additional similarity metrics can be plugged in via the
  ``metric`` keyword argument.
* **Batched** evaluation helper :pyfunc:`compute_score_batched` for efficient
  batch processing.

Example
~~~~~~~
>>> from verl.utils.reward_score.lexical import compute_score
>>> # Default token_ratio metric
>>> compute_score(
...     data_source="dummy",  # ignored
...     solution_str="Cats are great pets.",
...     ground_truth="Cats make wonderful companions."
... )
0.333...

>>> # Using ordered_token to consider token order
>>> compute_score(
...     data_source="dummy",
...     solution_str="the quick brown fox",
...     ground_truth="quick brown fox jumps",
...     metric="ordered_token"
... )
0.428...
"""

import pdb; # * Hacky way to debug the verl codebase (ray cluster)

from typing import Callable, List
import re
import warnings
import math

from difflib import SequenceMatcher

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    _HAS_BM25 = True
except ModuleNotFoundError:  # pragma: no cover – runtime fallback
    _HAS_BM25 = False
    warnings.warn(
        "rank-bm25 is not installed – falling back to SequenceMatcher. "
        "Install it via `pip install rank-bm25` for better lexical rewards.",
        RuntimeWarning,
    )

__all__: List[str] = [
    "compute_score",
    "compute_score_batched",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Tokenise *text* into a list of lowercase terms.
    
    Split on whitespace and punctuation, keeping alphanumeric tokens.
    This ensures single letters like 'A', 'B', 'C' are captured correctly.
    """
    # Use regex to split on non-alphanumeric characters and extract tokens
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def _token_ratio_score(query: str, document: str) -> float:
    """Compute token overlap ratio between query and document.
    
    Returns the ratio of common tokens to the total unique tokens in both texts.
    Uses Jaccard similarity: |A ∩ B| / |A ∪ B|
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)
    
    if not query_tokens and not doc_tokens:
        return 1.0  # Both empty = perfect match
    
    if not query_tokens or not doc_tokens:
        return 0.0  # One empty = no match
    
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    common_tokens = query_set & doc_set
    union_tokens = query_set | doc_set
    
    if not union_tokens:  # Should never happen, but safety check
        return 0.0
    
    # Jaccard similarity: intersection over union
    return len(common_tokens) / len(union_tokens)


def _token_precision_score(query: str, document: str) -> float:
    """Compute what fraction of query tokens appear in the document.
    
    This measures how well the document covers the query's content.
    Precision = |A ∩ B| / |A| where A is query tokens
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    common_tokens = query_set & doc_set
    
    # Precision: what fraction of query tokens are in document
    return len(common_tokens) / len(query_set)


def _token_recall_score(query: str, document: str) -> float:
    """Compute what fraction of document tokens appear in the query.
    
    This measures how much of the document is captured by the query.
    Recall = |A ∩ B| / |B| where B is document tokens
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    common_tokens = query_set & doc_set
    
    # Recall: what fraction of document tokens are in query
    return len(common_tokens) / len(doc_set)


def _token_f1_score(query: str, document: str) -> float:
    """Compute F1 score based on token precision and recall."""
    precision = _token_precision_score(query, document)
    recall = _token_recall_score(query, document)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def _ordered_token_score(query: str, document: str) -> float:
    """Compute similarity considering token order using longest common subsequence.
    
    This captures both token overlap and their relative ordering.
    Score = |LCS| / max(|A|, |B|) where LCS is longest common subsequence
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    # Compute LCS length
    m, n = len(query_tokens), len(doc_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if query_tokens[i-1] == doc_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    # Normalize by the longer sequence length
    # This ensures score is 1.0 only when sequences are identical
    max_length = max(m, n)
    return lcs_length / max_length if max_length > 0 else 0.0


def _bm25_score(query: str, document: str) -> float:
    """Compute BM25 score between query and document.
    
    Note: This is kept for reference but not recommended for reconstruction tasks.
    """
    if not _HAS_BM25:
        return _ratio_score(query, document)

    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    # Check for any overlap first
    if not set(query_tokens) & set(doc_tokens):
        return 0.0
    
    # Build BM25 index with single document
    bm25 = BM25Okapi([doc_tokens])
    raw_score = bm25.get_scores(query_tokens)[0]
    
    # Sigmoid normalization to map to [0, 1]
    return 1.0 / (1.0 + math.exp(-0.1 * raw_score))


def _ratio_score(a: str, b: str) -> float:
    """Simple character-based similarity using difflib (0–1)."""

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


_METRIC_FUNCS: dict[str, Callable[[str, str], float]] = {
    "token_ratio": _token_ratio_score,
    "token_precision": _token_precision_score,
    "token_recall": _token_recall_score,
    "token_f1": _token_f1_score,
    "ordered_token": _ordered_token_score,
    "bm25": _bm25_score,
    "ratio": _ratio_score,  # Character-based ratio
}

# -----------------------------------------------------------------------------
# Utility: filter reference list based on *extra_info*
# -----------------------------------------------------------------------------


def _filter_refs(refs: List[str], extra_info: dict | None) -> List[str]:
    """Return a possibly reduced list of *refs* according to *extra_info*.

    Supported options in *extra_info*:
    • ``target_gt`` – a string or list of strings; keep only references that exactly match any of them.
    • ``filter_gt_by_last_prompt_token`` (bool) **and** ``prompt`` – extract the
      last whitespace‐delimited token of *prompt* (lower‐cased) and keep only
      references that contain that token (after simple regex tokenisation).

    If filtering removes **all** references, the original list is returned so
    that scoring never fails with an empty pool.
    """

    if not extra_info or not isinstance(extra_info, dict):
        return refs

    # 1. Exact target string(s)
    # pdb.set_trace()
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
            subset = [r for r in refs if last_tok in _tokenize(r)]
            if subset:
                return subset

    return refs

# -----------------------------------------------------------------------------
# Optional Levenshtein distance metric (normalised similarity)
# -----------------------------------------------------------------------------

try:
    import Levenshtein  # type: ignore

    def _levenshtein_sim(a: str, b: str) -> float:  # noqa: D401
        """Normalised Levenshtein similarity (1‒dist/max_len)."""

        if not a and not b:
            return 1.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        dist = Levenshtein.distance(a, b)
        return max(0.0, 1.0 - dist / max_len)

    _METRIC_FUNCS["levenshtein"] = _levenshtein_sim  # type: ignore[assignment]
except ModuleNotFoundError:  # pragma: no cover
    warnings.warn(
        "python-Levenshtein not installed – the 'levenshtein' metric will be "
        "unavailable. Install it via `pip install python-Levenshtein`.",
        RuntimeWarning,
    )

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
    metric: str = "token_ratio",
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
        Extra information (currently unused).
    metric
        The similarity metric to use. Available options:
        - **token_ratio** (default): Jaccard similarity of tokens (intersection/union)
        - **token_precision**: Fraction of query tokens found in document
        - **token_recall**: Fraction of document tokens found in query
        - **token_f1**: F1 score combining precision and recall
        - **ordered_token**: Considers token order using longest common subsequence
        - **bm25**: BM25 score (not recommended for reconstruction tasks)
        - **ratio**: Character-based similarity

    Returns
    -------
    float
        A value in the range \[0, 1\] – higher means more similar.
    """

    # ------------------------------------------------------------------
    # Dispatch between *single* and *batch* calling conventions.
    # If list-based parameters are supplied (as done by BatchRewardManager), we
    # ignore the single-sample ones and return a list[float].
    # ------------------------------------------------------------------

    # Batch mode detection: BatchRewardManager passes plural-named params.
    if solution_strs is not None or ground_truths is not None:
        # Normalise lists (they should not be None by contract of caller)
        sol_list = solution_strs or []
        gt_list = ground_truths or []
        if not (len(sol_list) == len(gt_list)):
            # The API allows lengths to mismatch (one can be None). Align via
            # element-wise pairing up to min len; extras scored as 0.
            min_len = min(len(sol_list), len(gt_list))
            sol_list = sol_list[:min_len]
            gt_list = gt_list[:min_len]

        results: List[float] = []
        for sol, gt in zip(sol_list, gt_list):
            results.append(
                compute_score(
                    data_source=data_source,
                    solution_str=sol,
                    ground_truth=gt,
                    extra_info=None,
                    metric=metric,
                )
            )
        return results

    # ---------------- Single sample path ----------------

    if metric not in _METRIC_FUNCS:
        raise ValueError(
            f"Unknown lexical metric '{metric}'. Available: {list(_METRIC_FUNCS)}"
        )

    measure_fn = _METRIC_FUNCS[metric]

    # Non-batched mode: *solution_str* should be a single string, whereas
    # *ground_truth* can be a *list* of candidate references.  We therefore
    # treat ``ground_truth`` as an iterable of references regardless of its
    # original type.

    # Guard against None or non-string inputs for stability
    if not isinstance(solution_str, str) or solution_str == "":
        return 0.0

    solutions: List[str] = [solution_str]  # Expect exactly one entry
    refs: List[str] = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth or [])
    refs = _filter_refs(refs, extra_info)

    if not solutions or not refs:
        return 0.0

    # For all metrics, use the simple approach
    # No need for special caching since our metrics are already efficient

    per_sol_best = []
    for sol in solutions:
        best = 0.0
        for ref in refs:
            best = max(best, measure_fn(sol, ref))
            if best == 1.0:
                break
        per_sol_best.append(best)
    
    return sum(per_sol_best) / len(per_sol_best)


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str | List[str]],
    extra_infos: List[dict | None] | None = None,
    *,
    metric: str = "token_ratio",
):
    """Vectorised version of :pyfunc:`compute_score`.

    The implementation is intentionally simple – iterate over the inputs and
    call :pyfunc:`compute_score`. This keeps the example easy to follow. For
    maximum performance, consider pre-building BM25 indices for repeated
    references.
    """

    # Compile a single list of reference strings to compare against (ground_truths
    # may itself contain nested lists – flatten for convenience).
    flattened_refs: List[str] = []
    for gt in ground_truths:
        if isinstance(gt, list):
            flattened_refs.extend(gt)
        else:
            flattened_refs.append(gt)

    if not flattened_refs:
        # No references – return zeros to indicate no lexical grounding.
        return [0.0 for _ in solution_strs]

    results: List[float] = []
    defaults = [None] * len(solution_strs) if extra_infos is None else extra_infos

    # Re-use measure functions to avoid re-building BM25 index repeatedly when
    # metric is BM25 – pre-build once.
    measure_fn = _METRIC_FUNCS.get(metric)
    if measure_fn is None:
        raise ValueError(f"Unknown lexical metric '{metric}'.")

    needs_filter = False
    if extra_infos is not None:
        for ei in extra_infos:
            if isinstance(ei, dict) and (
                "target_gt" in ei or ei.get("filter_gt_by_prompt_token")
            ):
                needs_filter = True
                break

    # No special optimization needed for non-BM25 metrics
    # They are already simple and efficient

    if not needs_filter:
        for ds, sol, ei in zip(data_sources, solution_strs, defaults):
            best = 0.0
            for ref in flattened_refs:
                best = max(best, measure_fn(sol, ref))
                if best == 1.0:
                    break
            results.append(best)
        return results

    # Slow path: delegate to single-sample scorer so that per-example filtering applies.
    for ds, sol, gt, ei in zip(data_sources, solution_strs, ground_truths, defaults):
        results.append(
            compute_score(
                data_source=ds,
                solution_str=sol,
                ground_truth=gt,
                extra_info=ei,
                metric=metric,
            )
        )
 
    return results 