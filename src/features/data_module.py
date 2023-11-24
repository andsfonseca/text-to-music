import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .pre_processor import PreProcessor
from ..utils.data import TorchDataset


class DataModule(pl.LightningDataModule):
    """
    DataModule
    ===========

    Create a PyTorchLightningDataModule of a PreProcessor, this code prepare and download the data.
    This is the recommended approach to use with ddp instances

    Args:
        preprocessor (list[PreProcessor] | PreProcessor): A list of PreProcessor
        path (str): The path to save the preprocessed data
        batch_size (int): The batch size.
        transformation (function | list[function]): The function that will be applied to the dataset.
    """

    def __init__(self, preprocessors: list[PreProcessor] | PreProcessor, path: str, batch_size: int, transformations: list = None):
        super().__init__()
        
        if (type(preprocessors) != list):
            preprocessors = [preprocessors]

        if (type(transformations) != list):
            transformations = [transformations]

        self.preprocessors = preprocessors
        self.path = path
        self.batch_size = batch_size
        self.transformation = transformations

    def prepare_data(self):
        for preprocessor in self.preprocessors:
            preprocessor.get_train_test_split(
                path=self.path,
                save_split_sets=True)

    def setup(self, stage=None):

        self.train_datasets = []
        self.test_datasets = []

        for i, preprocessor in enumerate(self.preprocessors):
            train, test = preprocessor.get_train_test_split(
                path=self.path,
                save_split_sets=False,
                verbose=False)

            self.train_datasets.append(TorchDataset(
                train, transform=self.transformation[i]))
            self.test_datasets.append(TorchDataset(
                test, transform=self.transformation[i]))

    def train_dataloader(self):
        if len(self.preprocessors) == 1:
            return DataLoader(self.train_datasets[0], batch_size=self.batch_size)
        else:
            return [DataLoader(dataset, batch_size=self.batch_size) for dataset in self.train_datasets]

    def test_dataloader(self):
        if len(self.preprocessors) == 1:
            return DataLoader(self.test_datasets[0], batch_size=self.batch_size)
        else:
            return [DataLoader(dataset, batch_size=self.batch_size) for dataset in self.test_datasets]
