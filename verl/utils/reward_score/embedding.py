"""
Embedding similarity–based reward functions for VERL.

This module provides an easy way to compute a dense reward based on the
semantic similarity between the model response (``solution_str``) and the
reference answer (``ground_truth``).  The default backend is **FastText** –
each sentence is represented as the average of its token embeddings and the
reward is the cosine similarity (mapped to the range [0, 1]).

Highlights
----------
* **FastText** is used by default via either ``gensim``’s downloader
  (``fasttext-wiki-news-subwords-300``) *or* the official FastText Python
  package if you point the ``FASTTEXT_MODEL`` env-var to a ``.bin`` model.
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
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import List, Iterable, Tuple

import numpy as np

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


def _tokenise(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1024)
def _word_vec(word: str) -> np.ndarray:
    """Return embedding vector for *word* or a zero-vector if OOV / model missing."""

    if _MODEL is None:
        return np.zeros(_EMBED_DIM, dtype=np.float32)
    # Support both the official fastText API and gensim KeyedVectors fallback
    try:
        if hasattr(_MODEL, "get_word_vector"):
            return _MODEL.get_word_vector(word)  # type: ignore[attr-defined]
        # gensim KeyedVectors expose vectors via __getitem__ or get_vector
        if hasattr(_MODEL, "get_vector"):
            return _MODEL.get_vector(word)  # type: ignore[attr-defined]
        return _MODEL[word]  # type: ignore[index]
    except (KeyError, Exception):  # noqa: BLE001
        return np.zeros(_EMBED_DIM, dtype=np.float32)


def _sent_emb(sentence: str) -> np.ndarray:
    tokens = _tokenise(sentence)
    if not tokens or _MODEL is None:
        return np.zeros(_EMBED_DIM, dtype=np.float32)
    vecs = [_word_vec(tok) for tok in tokens]
    return np.mean(vecs, axis=0)


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

    tgt = extra_info.get("target_gt")
    if isinstance(tgt, str):
        subset = [r for r in refs if r == tgt]
        if subset:
            return subset

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


def _single_similarity(sol: str, ref: str) -> float:
    if _MODEL is None:
        return _lexical_ratio(sol, ref)
    return _cosine(_sent_emb(sol), _sent_emb(ref))


def _best_similarity(sol: str, refs: Iterable[str]) -> float:
    return max((_single_similarity(sol, r) for r in refs), default=0.0)


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
    """

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
            return [_best_similarity(s, gts_flat) for s in sols]

        # per-sample path with filtering
        res: List[float] = []
        defaults = [None] * len(sols) if extra_infos is None else extra_infos
        for sol, gt, ei in zip(sols, ground_truths, defaults):
            refs = [gt] if isinstance(gt, str) else list(gt)
            refs = _filter_refs(refs, ei)
            res.append(_best_similarity(sol, refs))
        return res

    # ---------------- Single sample path ----------------

    if solution_str is None or ground_truth is None:
        return 0.0

    refs = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    refs = _filter_refs(refs, extra_info)
    return _best_similarity(str(solution_str), refs)


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
::

    pip install fasttext gensim numpy

(Optional) Download a full‐sized model and set the environment variable::

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
Add to each example’s *extra_info* field::

    {"metric": "embedding"}

(the adapter already passes this through).

4. Extending / replacing the backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to use a different sentence model (e.g. Sentence-Transformers):

* Replace *_sent_emb* with your own embedding computation.
* Ensure the vector dimension is reflected in ``_EMBED_DIM``.
* Everything else (normalisation, batching) remains unchanged.
""" 