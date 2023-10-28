from datasets.arrow_dataset import Dataset

import torch


class Extractor(torch.utils.data.Dataset):
    """
    Extractor
    =======

    A abstract class that extracts the information from a dataset

    Args:
        dataset (Dataset): The dataset that contains the audio.
        column (str): The name of the audio column
        name (str): The name of the extractor. Default: "EmptyExtractor".
    """

    def __init__(self, dataset: Dataset, column: str, name="Extractor"):
        """
        Initializes a new Extractor
        """
        self.name = name
        self.dataset = dataset
        self.column = column

        # Define the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the data
        item = self.dataset[idx][self.column]
        return item
