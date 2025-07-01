import argparse
import os

import datasets

from verl.utils.fs import copy, makedirs  # type: ignore


def transform_example(example, idx: int, split: str):
    """Convert raw record into verl RL parquet compatible format."""
    return {
        'data_source': example.get('data_source', 'lexical_match_custom'),
        'prompt': [{'role': 'user', 'content': example.get('prompt', '')}],
        'ability': 'lexical_match',
        'reward_model': {
            'style': 'model',
            'ground_truth': example['ground_truths'],  # keep list
        },
        'extra_info': {
            'split': split,
            'index': idx,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Convert custom string-match HF dataset to verl parquet format.')
    parser.add_argument('--hf_dataset_dir', required=True, help='Path to the dataset directory produced by generate_lexical_match_dataset.py')
    parser.add_argument('--output_dir', default='~/data/lexical_match_custom/rl', help='Where to save parquet file')
    parser.add_argument('--hdfs_dir', default=None, help='Optional HDFS path to mirror parquet file')
    args = parser.parse_args()

    dataset = datasets.load_from_disk(os.path.expanduser(args.hf_dataset_dir))

    # Map transformation
    dataset = dataset.map(lambda ex, idx: transform_example(ex, idx, 'train'), with_indices=True, remove_columns=dataset.column_names)

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