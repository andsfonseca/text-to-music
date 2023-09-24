from abc import ABC, abstractmethod
from pathlib import Path

DATA_ROOT_FOLDER = "data"

class Dataset(ABC):

    def __init__(self, name = "EmptyDataset"):
        self.name = name
        self.dataset_dir = f"{DATA_ROOT_FOLDER}/{name}"
    
    @abstractmethod
    def generate(self):
        dir = Path(self.dataset_dir)
        dir.mkdir(exist_ok=True, parents=True)
