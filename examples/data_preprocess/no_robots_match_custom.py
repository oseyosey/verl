import argparse
import os
import json
import random
from functools import partial
from typing import List, Optional, Tuple

import datasets

from verl.utils.fs import copy, makedirs  # type: ignore


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


def transform_example(
    example,
    idx: int,
    split: str,
    original_id: int,
    match_type: str = "lexical",
    metric: str = "bm25",
    include_target_gt: bool = False,
    verbose: bool = True,
):
    """Convert a single No Robots record into verl RL parquet compatible format.

    We treat the conversation up to (but excluding) the last assistant message
    as the prompt, and the final assistant message as the ground-truth response.
    """

    if match_type not in {"lexical", "embedding"}:
        raise ValueError(f"Unsupported match_type: {match_type!r}. Choose 'lexical' or 'embedding'.")

    # Compose extra_info metadata
    extra_info = {
        "split": split,
        "index": idx,
        "metric": metric,
        # Provenance
        "id": original_id,  # original index from the split
        "prompt_id": example.get("prompt_id", None),
        "category": example.get("category", None),
    }

    # Select data_source based on match type
    if match_type == "lexical":
        data_source = "lexical_match_custom"
    else:  # match_type == "embedding"
        data_source = "embedding_match_custom"

    messages = example.get("messages", [])
    parsed = _find_last_assistant_and_prompt(messages, verbose=verbose and idx < 3)
    # Parsed should not be None if we filtered with _is_valid_example
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


def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert HuggingFaceH4/no_robots dataset to verl RL parquet format "
            "by sampling equally from train and test splits, with optional MIA JSONL outputs."
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
        help="If set, include the ground-truth solution as 'target_gt' in extra_info.",
    )
    parser.add_argument(
        "--subset_per_split",
        type=int,
        default=500,
        help=(
            "Number of examples to randomly subsample from each split (train and test). "
            "Will be clamped to the available split size."
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
        default="~/data/no-robots_match_custom/rl",
        help="Directory where the output Parquet file and optional JSONL files will be saved.",
    )
    parser.add_argument(
        "--output_name",
        default="no_robots",
        help="Base name used for JSONL outputs when --mia is set.",
    )
    parser.add_argument(
        "--mia",
        action="store_true",
        help="If set, also write MIA JSONL files for members (train) and nonmembers (test).",
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

    # Load splits
    ds_train_full = datasets.load_dataset("HuggingFaceH4/no_robots", split="train")
    ds_test_full = datasets.load_dataset("HuggingFaceH4/no_robots", split="test")

    # Filter to valid conversational examples
    ds_train_valid = ds_train_full.filter(_is_valid_example)
    ds_test_valid = ds_test_full.filter(_is_valid_example)

    if args.verbose:
        print(f"[main] Train valid size: {len(ds_train_valid)}; Test valid size: {len(ds_test_valid)}")

    # Determine per-split sample size
    k_train = min(args.subset_per_split, len(ds_train_valid))
    k_test = min(args.subset_per_split, len(ds_test_valid))
    if args.verbose and (k_train < args.subset_per_split or k_test < args.subset_per_split):
        print(
            f"[main] subset_per_split clamped to train={k_train}, test={k_test} due to split sizes."
        )

    rng = random.Random(args.seed)
    train_indices = sorted(rng.sample(range(len(ds_train_valid)), k_train)) if k_train < len(ds_train_valid) else list(range(len(ds_train_valid)))
    test_indices = sorted(rng.sample(range(len(ds_test_valid)), k_test)) if k_test < len(ds_test_valid) else list(range(len(ds_test_valid)))

    ds_train_sel = ds_train_valid.select(train_indices)
    ds_test_sel = ds_test_valid.select(test_indices)

    if args.verbose:
        print(f"[main] Selected train={len(ds_train_sel)}; test={len(ds_test_sel)}")

    # Prepare transformation closures that carry original indices from the full split
    # Note: we define original_id as the index within the valid-filtered split (pre-selection),
    # which corresponds to the original example position after filtering. If absolute indices
    # before filtering are needed, adjust to track mapping before filtering.
    transform_train = partial(
        transform_example,
        split="train",
        match_type=args.match_type,
        metric=args.metric,
        include_target_gt=args.include_target_gt,
        verbose=args.verbose,
    )

    transform_test = partial(
        transform_example,
        split="test",
        match_type=args.match_type,
        metric=args.metric,
        include_target_gt=args.include_target_gt,
        verbose=args.verbose,
    )

    # Map with original IDs preserved via the selected index mapping
    def _map_train(ex, i):
        original_id = train_indices[i]
        return transform_train(ex, idx=i, original_id=original_id)

    def _map_test(ex, i):
        original_id = test_indices[i]
        return transform_test(ex, idx=i, original_id=original_id)

    train_out = ds_train_sel.map(_map_train, with_indices=True, remove_columns=ds_train_sel.column_names)
    test_out = ds_test_sel.map(_map_test, with_indices=True, remove_columns=ds_test_sel.column_names)

    transformed = datasets.concatenate_datasets([train_out, test_out])
    
    # Shuffle to avoid ordering bias (deterministic with seed)
    transformed = transformed.shuffle(seed=args.seed)

    # ------------------------------------------------------------
    # Statistics: word counts for user (prompt) and assistant (ground truth)
    # ------------------------------------------------------------
    if args.verbose:
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
            print(f"  - user:      avg={avg_user:.2f} words, max={max_user} words")
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

    # Optionally write MIA JSONL files containing the sampled raw messages
    if args.mia:
        members_path = os.path.join(output_dir, f"{args.output_name}_members.jsonl")
        nonmembers_path = os.path.join(output_dir, f"{args.output_name}_nonmembers.jsonl")

        # Build rows with original IDs and raw messages
        members_rows = []
        for i in range(len(ds_train_sel)):
            original_id = train_indices[i]
            ex = ds_train_sel[i]
            row = {
                "id": original_id,
                "messages": ex.get("messages", []),
            }
            members_rows.append(row)

        nonmembers_rows = []
        for i in range(len(ds_test_sel)):
            original_id = test_indices[i]
            ex = ds_test_sel[i]
            row = {
                "id": original_id,
                "messages": ex.get("messages", []),
            }
            nonmembers_rows.append(row)

        _write_jsonl(members_path, members_rows)
        _write_jsonl(nonmembers_path, nonmembers_rows)
        print(f"Wrote MIA JSONL files: {members_path}, {nonmembers_path}")


if __name__ == "__main__":
    main()


