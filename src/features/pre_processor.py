from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.arrow_dataset import Dataset


class PreProcessor():
    """
    PreProcessor
    ===========

    Applys the pre-processing function and generate the split of the dataset.

    Args:
        dataset (Dataset): The dataset to be used.
        transformation_function (function): The function that will be applied to the dataset.
    """

    def __init__(self, dataset: Dataset, transformation_function: Callable):
        """
        Initializes a new PreProcessor.
        """

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = dataset
        self.transformation = transformation_function

    def perform(self):
        """
        Perform the transformation on the dataset and return the result.

        Returns:
            The result of the transformation.
        """
        return self.transformation(self.dataset)

    def get_train_test_split(self, test_size=0.2, random_state=78563, path: str = "", save_split_sets=True):
        """
        Generates train/test indexes and splits the dataset into train and test subsets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int): The seed used by the random number generator. Defaults to 78563.
            path (str): The path to save the train/test indexes and splits. Defaults to "".
            save_splt_sets (bool): Indicates whether to save the train/test splits. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The train and test sets as numpy arrays.
        """

        train_idx_path = Path(f"{path}/train_{random_state}_{test_size}.npy")
        test_idx_path = Path(f"{path}/test_{random_state}_{test_size}.npy")

        if (len(path) == 0 or not train_idx_path.exists()):
            print("Generating train/test indexes...")

            train_idx, test_idx = train_test_split(
                list(range(len(self.dataset))), test_size=test_size, random_state=random_state)

            if (len(path) != 0):
                print("Saving train/test indexes...")

                train_idx_path.parent.mkdir(exist_ok=True, parents=True)
                np.save(train_idx_path.absolute(), train_idx)
                np.save(test_idx_path.absolute(), test_idx)

        else:
            print("Loading train/test indexes...")

            train_idx = np.load(train_idx_path.absolute(), )
            test_idx = np.load(test_idx_path.absolute())

        transformed_data = self.perform()

        train_subset_path = Path(
            f"{path}/train_{random_state}_{test_size}_{transformed_data.name.lower()}.npy")
        test_subset_path = Path(
            f"{path}/test_{random_state}_{test_size}_{transformed_data.name.lower()}.npy")

        if (len(path) == 0 or not train_subset_path.exists()):

            train_subset = Subset(transformed_data, train_idx if type(
                train_idx) == list else train_idx.tolist())
            train_data = np.array([data.to(self.device).cpu().numpy() for data in tqdm(
                train_subset, desc=f"Generating train subset [{transformed_data.name}]")])

            test_subset = Subset(transformed_data, test_idx if type(
                test_idx) == list else test_idx.tolist())
            test_data = np.array([data.to(self.device).cpu().numpy() for data in tqdm(
                test_subset, desc=f"Generating test subset [{transformed_data.name}]")])

            if (len(path) != 0 and save_split_sets):
                print("Saving train/test splits...")

                train_subset_path.parent.mkdir(exist_ok=True, parents=True)
                np.save(train_subset_path.absolute(), train_data)
                np.save(test_subset_path.absolute(), test_data)

        else:
            print("Loading train/test splits...")

            train_data = np.load(train_subset_path.absolute())
            test_data = np.load(test_subset_path.absolute())

        return train_data, test_data
