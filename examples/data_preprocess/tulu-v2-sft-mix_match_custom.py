import argparse
import os
import json
import random
from functools import partial
from typing import List, Optional, Tuple

import datasets

from verl.utils.fs import copy, makedirs  # type: ignore


def _parse_include_datasets(raw: Optional[str]) -> List[str]:
    if raw is None or raw.strip() == "":
        # No filtering when not specified
        return []
    # Support comma-separated values and/or repeated flags by letting argparse
    # pass a comma-joined string when action="append". Normalize here.
    parts = []
    for chunk in raw.split(","):
        name = chunk.strip()
        if name:
            parts.append(name)
    return parts


def _find_last_assistant_and_prompt(messages: List[dict], verbose: bool = False) -> Optional[Tuple[List[dict], str]]:
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        role = messages[i].get("role", "")
        if role == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        # No assistant response found or it's the very first message (nothing to prompt with)
        return None

    ground_truth = str(messages[last_assistant_idx].get("content", "")).strip()
    prompt_msgs = messages[:last_assistant_idx]

    # Ensure each message has the expected structure
    normalized_prompt = []
    for m in prompt_msgs:
        role = str(m.get("role", "user"))
        content = str(m.get("content", "")).strip()
        if content == "":
            # Skip empty content to avoid degenerate prompts
            continue
        normalized_prompt.append({"role": role, "content": content})

    if len(normalized_prompt) == 0 or ground_truth == "":
        return None

    if verbose:
        preview = {
            "prompt": normalized_prompt[-2:] if len(normalized_prompt) > 2 else normalized_prompt,
            "ground_truth_preview": ground_truth[:120],
        }
        print("[_find_last_assistant_and_prompt] Preview:")
        print(json.dumps(preview, indent=2, ensure_ascii=False))

    return normalized_prompt, ground_truth


def _is_valid_example(example) -> bool:
    messages = example.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role", "") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        return False
    gt = str(messages[last_assistant_idx].get("content", "")).strip()
    if gt == "":
        return False
    # Ensure there is at least one non-empty prior message
    for m in messages[:last_assistant_idx]:
        c = str(m.get("content", "")).strip()
        if c:
            return True
    return False


def transform_example(example, idx: int, split: str, match_type: str = "lexical", metric: str = "bm25", include_target_gt: bool = False, verbose: bool = True):
    """Convert a single Tulu v2 SFT mixture record into verl RL parquet compatible format.

    The record contains conversational messages. We treat the conversation up to
    (but excluding) the last assistant message as the prompt, and the final
    assistant message as the ground-truth response to be matched by the reward
    function (lexical or embedding).
    """

    if match_type not in {"lexical", "embedding"}:
        raise ValueError(f"Unsupported match_type: {match_type!r}. Choose 'lexical' or 'embedding'.")

    # Compose extra_info metadata
    extra_info = {
        "split": split,
        "index": idx,
        "metric": metric,
        # Keep provenance for analysis
        "dataset": example.get("dataset", None),
        "id": example.get("id", None),
    }

    # Optionally include the ground-truth answer also as 'target_gt'
    # (some reward functions may leverage this hint)
    # ground_truth is determined below from messages

    # Select data_source based on match type
    if match_type == "lexical":
        data_source = "lexical_match_custom"
    else:  # match_type == "embedding"
        data_source = "embedding_match_custom"

    messages = example.get("messages", [])
    parsed = _find_last_assistant_and_prompt(messages, verbose=verbose and idx < 3)
    # By construction, we filtered invalid examples beforehand; parsed should not be None
    assert parsed is not None, "Unexpected invalid example encountered after filtering."
    prompt_messages, ground_truth = parsed

    if include_target_gt:
        extra_info["target_gt"] = ground_truth

    record = {
        "data_source": data_source,
        "prompt": prompt_messages,
        "ability": "lexical_match",
        "reward_model": {
            "style": "model",
            "ground_truth": ground_truth,
        },
        "extra_info": extra_info,
    }

    if verbose and idx < 3:
        print("[transform_example] Preview of transformed record (truncated):")
        print(json.dumps(record, indent=2, ensure_ascii=False)[:1000])

    return record


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert allenai/tulu-v2-sft-mixture dataset to verl RL parquet format "
            "(lexical or embedding match), with controllable random subsampling "
            "from specified source datasets (default: math-500)."
        )
    )
    parser.add_argument(
        "--match_type",
        choices=["lexical", "embedding"],
        default="lexical",
        help="Choose reward type: lexical (BM25 / ratio / etc.) or embedding similarity.",
    )
    parser.add_argument(
        "--metric",
        default="bm25",
        help="Similarity metric to store in extra_info (e.g. bm25, ratio, levenshtein, embedding).",
    )
    parser.add_argument(
        "--include_target_gt",
        action="store_true",
        help="If set, include the ground-truth solution as 'target_gt' in extra_info so that reward functions can filter references.",
    )
    parser.add_argument(
        "--include_datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated list of dataset names (values from the 'dataset' field) to filter from. "
        ),
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=500,
        help=(
            "Number of examples to randomly subsample from the filtered split. "
            "Default 500 (the full size of math-500). Use -1 to keep all available."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subsampling.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/tulu-v2-sft-mix_match_custom/rl",
        help="Directory where the output Parquet file will be saved.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to mirror the Parquet file to.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging the preprocessing pipeline.",
    )

    args = parser.parse_args()

    # Load the full dataset
    ds_full = datasets.load_dataset("allenai/tulu-v2-sft-mixture", split="train")

    # Normalize include_datasets into a list
    include_names = [s.lower() for s in _parse_include_datasets(args.include_datasets)]

    # Filter by the 'dataset' column only if user requested filtering
    if len(include_names) > 0 and "dataset" in ds_full.column_names:
        def _match_dataset(ex):
            src = str(ex.get("dataset", "")).lower()
            return any(k in src for k in include_names)

        ds_filtered = ds_full.filter(_match_dataset)
    else:
        if len(include_names) > 0 and "dataset" not in ds_full.column_names and args.verbose:
            print("[main] Warning: 'dataset' column not found; skipping dataset-name filtering.")
        ds_filtered = ds_full

    if args.verbose:
        print(f"[main] Filtered dataset size after dataset-name filter: {len(ds_filtered)}")

    # Subsample deterministically if requested
    if args.subset_size is not None and args.subset_size >= 0:
        target_k = args.subset_size
        n = len(ds_filtered)
        if target_k > n:
            if args.verbose:
                print(f"[main] Requested subset_size {target_k} exceeds available {n}. Using full {n}.")
            target_k = n
        if target_k < n:
            rng = random.Random(args.seed)
            indices = rng.sample(range(n), target_k)
            ds_filtered = ds_filtered.select(sorted(indices))
            if args.verbose:
                print(f"[main] Subsampled to {len(ds_filtered)} examples with seed {args.seed}.")
        else:
            if args.verbose:
                print(f"[main] Using all {n} filtered examples (no subsampling).")
    else:
        if args.verbose:
            print(f"[main] subset_size set to {args.subset_size}; keeping all {len(ds_filtered)} examples.")

    # Further filter to valid conversational examples that have a final assistant turn
    ds_filtered = ds_filtered.filter(_is_valid_example)

    # Prepare transformation function
    transform_fn = partial(
        transform_example,
        split="train",
        match_type=args.match_type,
        metric=args.metric,
        include_target_gt=args.include_target_gt,
        verbose=args.verbose,
    )

    # Map and remove original columns
    transformed = ds_filtered.map(transform_fn, with_indices=True, remove_columns=ds_filtered.column_names)

    # The mapping produced full records; there are no original columns to remove here.
    if args.verbose:
        print("\n[main] Finished transformation – preview of transformed record(s):")
        preview_recs = transformed[:3]
        print(json.dumps(preview_recs, indent=2, ensure_ascii=False))

        # ------------------------------------------------------------
        # Statistics: word counts for user (prompt) and assistant (ground truth)
        # ------------------------------------------------------------
        def _count_words(text: str) -> int:
            return len(str(text).strip().split())

        user_lengths: List[int] = []
        assistant_lengths: List[int] = []

        for rec in transformed:
            # Sum words across all 'user' messages in the prompt
            u_sum = 0
            for msg in rec.get("prompt", []):
                if msg.get("role") == "user":
                    u_sum += _count_words(msg.get("content", ""))
            # Words in assistant ground-truth
            a_len = _count_words(rec.get("reward_model", {}).get("ground_truth", ""))
            user_lengths.append(u_sum)
            assistant_lengths.append(a_len)

        if len(user_lengths) > 0:
            avg_user = sum(user_lengths) / len(user_lengths)
            max_user = max(user_lengths)
            avg_asst = sum(assistant_lengths) / len(assistant_lengths)
            max_asst = max(assistant_lengths)

            print("\n[stats] Word-count statistics over transformed dataset:")
            print(f"  - user:     avg={avg_user:.2f} words, max={max_user} words")
            print(f"  - assistant: avg={avg_asst:.2f} words, max={max_asst} words")

    # Persist to local disk.
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.parquet")
    transformed.to_parquet(out_path)
    print(f"Parquet file saved at {out_path}")

    # Optionally copy to HDFS.
    if args.hdfs_dir:
        hdfs_path = os.path.join(args.hdfs_dir, "train.parquet")
        makedirs(args.hdfs_dir)
        copy(out_path, hdfs_path)
        print(f"Also copied to HDFS: {hdfs_path}")


if __name__ == "__main__":
    main()


