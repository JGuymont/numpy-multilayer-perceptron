#!/usr/bin/env python
"""
Usage:
    python split_data.py --csv_path ./data/circles/circles.txt --delimiter \s --train_size 0.6 --valid_size 0.2 --test_size 0.2 --out_dir ./data/circles/
    python split_data.py --csv_path ./data/mnist/train.csv --train_size 0.7 --valid_size 0.3 --out_dir ./data/mnist/
"""
import argparse
import random
import pandas


def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--delimiter', type=str, default=",")
    parser.add_argument('--train_size', type=float, default=0.)
    parser.add_argument('--valid_size', type=float, default=0.)
    parser.add_argument('--test_size', type=float, default=0.)
    parser.add_argument('--out_dir', type=str)
    return parser.parse_args()


def load_data(path, delimiter):
    """
    Load data into a pandas dataframe
    """
    dataframe = pandas.read_csv(path, delimiter=delimiter)
    return dataframe


def get_split(args):
    assert args.train_size + args.valid_size + args.test_size == 1.
    split = {}
    if args.train_size > 0.:
        split['train'] = args.train_size
    if args.valid_size > 0.:
        split['valid'] = args.valid_size
    if args.test_size > 0.:
        split['test'] = args.test_size
    return split


def _get_subset_sizes(data_size, split):
    n_subset = len(split)
    subset_sizes = {}
    subset_counter = 0
    data_counter = 0
    for subset_name, subset_pct in split.items():
        subset_counter += 1
        if subset_counter == n_subset:
            subset_sizes[subset_name] = data_size - data_counter
        else:
            subset_sizes[subset_name] = round(data_size * subset_pct)
        assert abs(subset_sizes[subset_name] / data_size - subset_pct) < 0.005
        data_counter += subset_sizes[subset_name]
    assert sum(subset_sizes.values()) == data_size
    return subset_sizes


def _get_split_idx(data_size, split):
    examples = range(data_size)
    subset_sizes = _get_subset_sizes(data_size, split)
    split_idx = {}
    for subset_name, subset_size in subset_sizes.items():
        split_idx[subset_name] = random.sample(examples, subset_size)
        examples = [example for example in examples if example not in split_idx[subset_name]]
    return split_idx


def split_data(dataframe, split):
    data_size = len(dataframe)
    split_idx = _get_split_idx(data_size, split)
    data = {k: dataframe.iloc[v] for k, v in split_idx.items()}
    return data


def save_data(dataframe, name, out_dir):
    path = '{}{}.csv'.format(out_dir, name)
    dataframe.to_csv(path, index=False)


def main(args):
    split = get_split(args)
    data = load_data(args.csv_path, delimiter=args.delimiter)
    data = split_data(data, split)
    for data_name, data in data.items():
        save_data(data, data_name, args.out_dir)


if __name__ == '__main__':
    main(argparser())
