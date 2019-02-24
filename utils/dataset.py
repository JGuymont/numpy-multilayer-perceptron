import numpy as np
import pandas


class Dataset:
    """
    Abstract data class

    Args
        path (str): Path to the data file.
        input_features (list): List of the indices of the input features.
        output_features (list): List of indices of the output features.
    """

    def __init__(self, path, input_features, output_features, header=None, transform=None):
        self.data = pandas.read_csv(path, header=header)
        self.input_idx = input_features
        self.output_idx = output_features
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data.iloc[i]
        input = list(example[self.input_idx[0]:self.input_idx[1]])
        if self.transform:
            input = self.transform(input)
        target = int(example[self.output_idx])
        return input, target
