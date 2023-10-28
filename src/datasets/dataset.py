from abc import ABC, abstractmethod
from pathlib import Path

DATA_ROOT_FOLDER = "data"
"""Default folder for saving dataset assets
"""


class Dataset(ABC):
    """
    Dataset
    =======

    Abstract class of a generic dataset. Contains a simple interface for building a dataset
    generator

    Args:
        name (str): The name of the dataset. Default: EmptyDataset.
    """

    def __init__(self, name="EmptyDataset"):
        """Initializes a new Dataset
        """
        self.name = name
        self.dataset_dir = f"{DATA_ROOT_FOLDER}/{name}"

    @abstractmethod
    def generate(self):
        """
        Method responsible for generating the dataset. During generation, a folder is created with
        the name of the dataset within the path defined by the DATA_ROOT_FOLDER constant

        """
        dir = Path(self.dataset_dir)
        dir.mkdir(exist_ok=True, parents=True)

        return None

    def get_raw_folder(self):
        """
        Returns the path to the raw folder for the dataset.

        Returns:
            str: The path to the raw folder.
        """
        raw_dir = f"{self.dataset_dir}/raw"
        Path(raw_dir).mkdir(exist_ok=True, parents=True)
        return raw_dir

    def get_interim_folder(self):
        """
        Returns the path to the interim folder for the dataset.

        Returns:
            str: The path to the interim folder.
        """
        interim_dir = f"{self.dataset_dir}/interim"
        Path(interim_dir).mkdir(exist_ok=True, parents=True)
        return interim_dir

    def get_processed_folder(self):
        """
        Returns the path to the processed folder for the dataset.

        Returns:
            str: The path to the processed folder.
        """
        processed_dir = f"{self.dataset_dir}/processed"
        Path(processed_dir).mkdir(exist_ok=True, parents=True)
        return processed_dir
