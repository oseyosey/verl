import argparse
import os

import datasets

from verl.utils.fs import copy, makedirs  # type: ignore


def transform_example(example, idx: int, split: str, metric: str = 'bm25', verbose: bool = True):
    """Convert raw record into verl RL parquet compatible format.

    The *metric* argument specifies which lexical metric (bm25 / ratio / levenshtein)
    should be applied at training-time.  It is stored in ``extra_info`` so that the
    reward loader can forward it to ``lexical.compute_score``.
    """
    extra_info = {
        'split': split,
        'index': idx,
        'metric': metric,
    }
    if 'assistant_prefix' in example:
        if verbose:
            print(f"[transform_example] Adding assistant_prefix for idx={idx}: {example['assistant_prefix']}")
        extra_info['assistant_prefix'] = example['assistant_prefix']

    return {
        'data_source': example.get('data_source', 'embedding_match_custom'),
        'prompt': [{'role': 'user', 'content': example.get('prompt', '')}],
        'ability': 'lexical_match',
        'reward_model': {
            'style': 'model',
            'ground_truth': example['ground_truths'],  # keep list
        },
        'extra_info': extra_info,
    }


def main():
    parser = argparse.ArgumentParser(description='Convert custom string-match HF dataset to verl parquet format.')
    parser.add_argument('--hf_dataset_dir', required=True, help='Path to the dataset directory produced by generate_lexical_match_dataset.py')
    parser.add_argument('--output_dir', default='~/data/lexical_match_custom/rl', help='Where to save parquet file')
    parser.add_argument('--hdfs_dir', default=None, help='Optional HDFS path to mirror parquet file')
    parser.add_argument('--metric', choices=['embedding'], default='embedding', help='Embedding similarity metric to be used during RL training')
    args = parser.parse_args()

    dataset = datasets.load_from_disk(os.path.expanduser(args.hf_dataset_dir))

    # Map transformation with selected metric
    from functools import partial
    transform_fn = partial(transform_example, split='train', metric=args.metric)
    dataset = dataset.map(transform_fn, with_indices=True, remove_columns=dataset.column_names)

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'train.parquet')
    dataset.to_parquet(out_path)
    print(f'Parquet file saved at {out_path}')

    if args.hdfs_dir:
        hdfs_path = os.path.join(args.hdfs_dir, 'train.parquet')
        makedirs(args.hdfs_dir)
        copy(out_path, hdfs_path)
        print(f'Also copied to HDFS: {hdfs_path}')


if __name__ == '__main__':
    main() 