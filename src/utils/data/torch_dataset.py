import torch
from torch.utils.data import Dataset
import numpy as np


class TorchDataset(Dataset):
    """
    TorchDataset
    =======

    Create a dataset from an ArrayLike and apply a transformation funtion.

    Args:
        data (ArrayLike): The input data for the class. It can be either a numpy array or a torch tensor.
        transform (Callable): The transformation to apply to the data. Default: None
    """

    def __init__(self, data, transform=None):
        """
        Initialize the TorchDataset
        """

        # Convert the data to a torch tensor
        if isinstance(data, np.ndarray) and not data.dtype.kind in ['U', 'S']:
            data = torch.from_numpy(data)

        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        # Apply the transformation
        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)
