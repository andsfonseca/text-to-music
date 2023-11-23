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
        preprocessor (PreProcessor): The PreProcessor
        path (str): The path to save the preprocessed data
        batch_size (int): The batch size.
        transformation (function): The function that will be applied to the dataset.
    """

    def __init__(self, preprocessor: PreProcessor, path: str, batch_size : int, transformation = None):
        super().__init__()
        self.preprocessor = preprocessor
        self.path = path
        self.batch_size = batch_size
        self.transformation = transformation

    def prepare_data(self):
        self.preprocessor.get_train_test_split(
            path=self.path,
            save_split_sets=True)

    def setup(self, stage=None):
        train, test = self.preprocessor.get_train_test_split(
            path=self.path,
            save_split_sets=False,
            verbose=False)

        self.train_dataset = TorchDataset(
            train, transform=self.transformation)
        self.test_dataset = TorchDataset(
            test, transform=self.transformation)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
