import argparse
import os
from functools import partial

import datasets
import json

from verl.utils.fs import copy, makedirs  # type: ignore


def transform_example(example, idx: int, split: str, match_type: str = "lexical", metric: str = "bm25", include_target_gt: bool = False, verbose: bool = True):
    """Convert AIME 2025 record into verl RL parquet compatible format.

    Parameters
    ----------
    example : dict
        A single raw dataset record coming from *allenai/aime-2021-2025* (2025 subset).
    idx : int
        Index within the split – used to create a stable identifier.
    split : str
        Data split name (e.g. "train" or "test").  Currently always "train".
    match_type : {"lexical", "embedding"}
        Determines which reward type will be used at *training-time* and thus
        controls the ``data_source`` string expected by
        ``verl.utils.reward_score.default_compute_score``.
    metric : str
        Specific similarity metric to be applied by the reward function.  For
        lexical matching typical values are "bm25", "ratio" or "levenshtein";
        for embedding matching the default value "embedding" is recommended (it
        is simply forwarded to the embedding reward implementation).
    verbose : bool, default True
        Print extra information while transforming – useful for debugging.
    """

    if match_type not in {"lexical", "embedding"}:
        raise ValueError(f"Unsupported match_type: {match_type!r}. Choose 'lexical' or 'embedding'.")

    # Construct *extra_info* section – this is later accessible by the reward
    # loader so that it can forward *metric* to the scoring function.
    extra_info = {
        "split": split,
        "index": idx,
        "metric": metric,
    }

    # Optionally include the ground-truth answer as a dedicated target reference
    # for reward functions that support the 'target_gt' hint.
    if include_target_gt:
        if verbose:
            print(f"[transform_example] Added target_gt for idx={idx}: {str(example['solution']).strip()[:100]}")
        extra_info["target_gt"] = str(example["solution"]).strip()

    # Preserve optional assistant prefix if upstream pipeline inserted one.
    # NOT USED IN THIS DATASET.
    if "assistant_prefix" in example:
        if verbose:
            print(
                f"[transform_example] Adding assistant_prefix for idx={idx}: {example['assistant_prefix']}"
            )
        extra_info["assistant_prefix"] = example["assistant_prefix"]

    # Decide on *data_source* according to requested matching type.
    if match_type == "lexical":
        data_source = "lexical_match_custom"
    elif match_type == "embedding":  # match_type == "embedding"
        data_source = "embedding_match_custom"
    else:
        raise ValueError(f"Unsupported match_type: {match_type!r}. Choose 'lexical' or 'embedding'.")

    # Build the verl record.
    record = {
        "data_source": data_source,
        # Only include the *problem statement* – the solution will be generated
        # by the model during RL fine-tuning.
        "prompt": [{"role": "user", "content": str(example["problem"]).strip()}],
        # Ability tag is arbitrary – we reuse "lexical_match" for compatibility.
        "ability": "lexical_match",
        "reward_model": {
            "style": "model",
            # Ground-truth reference that the reward model will compare against.
            "ground_truth": str(example["solution"]).strip(),
        },
        "extra_info": extra_info,
    }

    # For quick sanity-checking: print the structure of the record for the first
    # few samples when *verbose* is enabled.
    if verbose and idx < 3:
        print("[transform_example] Preview of transformed record (truncated):")
        print(json.dumps(record, indent=2, ensure_ascii=False)[:1000])

    return record


def main():
    parser = argparse.ArgumentParser(
        description="Convert allenai/aime-2021-2025 dataset (2025 subset) to verl RL parquet format (lexical or embedding match)."
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
        "--output_dir",
        default="~/data/aime2025_match_custom/rl",
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help=(
            "Optional target dataset length. Must be greater than the original "
            "dataset size; records are duplicated cyclically to reach this size."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load dataset and select only 2025 entries (id 90–119 / last 30 rows)
    # ------------------------------------------------------------

    ds_full = datasets.load_dataset("allenai/aime-2021-2025", split="train")

    # Strategy 1: prefer filtering by explicit 'id' field if present
    if "id" in ds_full.column_names:
        ds = ds_full.filter(lambda x: 90 <= int(x["id"]) <= 119)
    else:
        # Fallback: simply take the last 30 examples
        ds = ds_full.select(range(len(ds_full) - 30, len(ds_full)))

    if args.verbose:
        print(f"[main] Loaded {len(ds)} samples for year 2025 from allenai/aime-2021-2025 dataset.")

    # Prepare transformation function with fixed parameters.
    transform_fn = partial(
        transform_example,
        split="train",
        match_type=args.match_type,
        metric=args.metric,
        include_target_gt=args.include_target_gt,
        verbose=args.verbose,
    )

    ds_transformed = ds.map(transform_fn, with_indices=True, remove_columns=ds.column_names)

    # ------------------------------------------------------------------
    # Optional up-sampling to reach desired number of samples
    # ------------------------------------------------------------------

    if args.num_samples is not None:
        if args.num_samples < len(ds_transformed):
            raise ValueError(
                "--num_samples must be greater than or equal to the original dataset size "
                f"({len(ds_transformed)}). Got {args.num_samples}."
            )
        if args.num_samples > len(ds_transformed):
            if args.verbose:
                print(
                    f"[main] Upsampling dataset from {len(ds_transformed)} to {args.num_samples} samples by cyclic duplication."
                )

            # Create cyclic index list
            indices = [i % len(ds_transformed) for i in range(args.num_samples)]
            ds_transformed = ds_transformed.select(indices)

    if args.verbose:
        print("\n[main] Finished transformation – preview of transformed record(s):")
        preview_recs = ds_transformed[:3]
        print(json.dumps(preview_recs, indent=2, ensure_ascii=False))

    # Persist to local disk.
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.parquet")
    ds_transformed.to_parquet(out_path)
    print(f"Parquet file saved at {out_path}")

    # Optionally copy to HDFS.
    if args.hdfs_dir:
        hdfs_path = os.path.join(args.hdfs_dir, "train.parquet")
        makedirs(args.hdfs_dir)
        copy(out_path, hdfs_path)
        print(f"Also copied to HDFS: {hdfs_path}")


if __name__ == "__main__":
    main()

