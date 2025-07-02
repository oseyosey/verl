from __future__ import annotations

"""Lexical similarity–based reward functions.

This module offers a simple way to evaluate the lexical similarity between the
model response (``solution_str``) and ground-truth answer (``ground_truth``).
The main entry-point is :pyfunc:`compute_score`, which follows the interface
expected by verl's reward loading utilities (see
`verl.trainer.ppo.reward.get_custom_reward_fn`).

Key features
------------
* **BM25** is used as the default scoring method (via the *rank-bm25* library).
* **Fallback** to `difflib.SequenceMatcher` when *rank-bm25* is not available.
* **Extensible** – additional similarity metrics can be plugged in via the
  ``metric`` keyword argument.
* **Batched** evaluation helper :pyfunc:`compute_score_batched` to showcase how
  to vectorise the computation (useful when RewardManager supports batching).

Example
~~~~~~~
>>> from verl.utils.reward_score.lexical import compute_score
>>> compute_score(
...     data_source="dummy",  # ignored
...     solution_str="Cats are great pets.",
...     ground_truth="Cats make wonderful companions."  # doctest: +ELLIPSIS
... )
0.7...

If you need a different metric (e.g. *ratio*):
>>> compute_score(
...     data_source="dummy",
...     solution_str="hello world",
...     ground_truth="hello, world!",
...     metric="ratio",
... )
1.0
"""

import pdb; # * Hacky way to debug the verl codebase (ray cluster)

from typing import Callable, List
import re
import warnings

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
    """Tokenise *text* into a list of lowercase terms suitable for BM25."""
    # Extract alphanumeric sequences as tokens
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _bm25_single_score(query: str, document: str) -> float:
    """Compute a *normalised* BM25 score of *query* against single *document*.

    The score is divided by the *ideal* score (query versus itself) to bound
    the result in \[0, 1\].
    """

    if not _HAS_BM25:
        # Fallback path – shouldn't normally happen when library is installed.
        return _ratio_score(query, document)

    # BM25Okapi expects a *corpus* (list of token lists). We build a tiny corpus
    # containing only the reference document to keep things simple.
    tokenised_corpus = [_tokenize(document)]
    bm25 = BM25Okapi(tokenised_corpus)

    query_tokens = _tokenize(query)
    raw_score = bm25.get_scores(query_tokens)[0]  # array of length 1

    # "Ideal" score: query compared to itself (upper bound)
    ideal_score = bm25.get_scores(_tokenize(document))[0]
    if ideal_score == 0:
        return 0.0
    return min(raw_score / ideal_score, 1.0)


def _ratio_score(a: str, b: str) -> float:
    """Simple character-based similarity using difflib (0–1)."""

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


_METRIC_FUNCS: dict[str, Callable[[str, str], float]] = {
    "bm25": _bm25_single_score,
    "ratio": _ratio_score,
}

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
    metric: str = "bm25",
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
        The similarity metric to use. **bm25** (default) and **ratio** are
        provided out-of-the-box. You can extend :pydata:`_METRIC_FUNCS` at
        runtime to register new metrics.

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

    solutions: List[str] = [solution_str]  # Expect exactly one entry
    refs: List[str] = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth or [])

    if not solutions or not refs:
        return 0.0

    # Build BM25 index once when needed (performance).
    if metric == "bm25" and _HAS_BM25:
        # Pre-tokenise the reference corpus and build a single BM25 index.
        docs_tokens = [_tokenize(r) for r in refs]
        bm25_index = BM25Okapi(docs_tokens)

        # Compute an *ideal* score for every reference – that is, the score
        # obtained when the query exactly matches the document itself.  This
        # provides a true upper-bound (≥ raw_score) for normalisation and
        # guarantees that raw_score / ideal_score is ≤ 1.0.
        ideal_scores = [
            bm25_index.get_scores(doc_tokens)[idx]
            for idx, doc_tokens in enumerate(docs_tokens)
        ]

        # NOTE: `compute_score` expects *measure_fn* callables to accept two
        # positional arguments (query and reference) even though the BM25
        # implementation can score the query against *all* references at
        # once via the pre-built index. We therefore include an extra, unused
        # positional parameter so that the signature matches the expected
        # ``Callable[[str, str], float]`` type and avoids ``TypeError`` when
        # invoked as ``measure_fn(sol, ref)`` inside the nested loop.

        def _bm25_cached(q: str, _unused_ref: str | None = None) -> float:  # noqa: D401
            """Return the best normalised BM25 score of *q* w.r.t the index.

            The additional ``_unused_ref`` parameter is present **only** to
            satisfy the two-argument calling convention used elsewhere in this
            module – it is ignored during computation.
            """

            q_tokens = _tokenize(q)
            raw_scores = bm25_index.get_scores(q_tokens)
            best = 0.0
            for rs, ideal in zip(raw_scores, ideal_scores):
                if ideal > 0:
                    best = max(best, rs / ideal)
            return min(best, 1.0)

        measure_fn = _bm25_cached

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
    metric: str = "bm25",
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
        flattened_refs = [""]  # Avoid zero-division / empty loops

    results: List[float] = []
    defaults = [None] * len(solution_strs) if extra_infos is None else extra_infos

    # Re-use measure functions to avoid re-building BM25 index repeatedly when
    # metric is BM25 – pre-build once.
    measure_fn = _METRIC_FUNCS.get(metric)
    if measure_fn is None:
        raise ValueError(f"Unknown lexical metric '{metric}'.")

    if metric == "bm25" and _HAS_BM25:
        docs_tokens = [_tokenize(r) for r in flattened_refs]
        bm25_index = BM25Okapi(docs_tokens)
        ideal_scores = [
            bm25_index.get_scores(doc_tokens)[idx]
            for idx, doc_tokens in enumerate(docs_tokens)
        ]

        # NOTE: `compute_score` expects *measure_fn* callables to accept two
        # positional arguments (query and reference) even though the BM25
        # implementation can score the query against *all* references at
        # once via the pre-built index. We therefore include an extra, unused
        # positional parameter so that the signature matches the expected
        # ``Callable[[str, str], float]`` type and avoids ``TypeError`` when
        # invoked as ``measure_fn(sol, ref)`` inside the nested loop.

        def bm25_batch(q: str, _unused_ref: str | None = None) -> float:  # noqa: D401
            """BM25 scoring helper matching the two-argument call signature."""

            q_tokens = _tokenize(q)
            raw_scores = bm25_index.get_scores(q_tokens)
            best = 0.0
            for rs, ideal in zip(raw_scores, ideal_scores):
                if ideal > 0:
                    best = max(best, rs / ideal)
                if best == 1.0:
                    break
            return best

        measure_fn = bm25_batch

    for ds, sol, ei in zip(data_sources, solution_strs, defaults):
        best = 0.0
        for ref in flattened_refs:
            best = max(best, measure_fn(sol, ref))
            if best == 1.0:
                break
        results.append(best)

    return results 