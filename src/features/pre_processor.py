from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets.arrow_dataset import Dataset

from ..utils.gpu import create_device

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

        self.device = create_device()

        self.dataset = dataset
        self.transformation = transformation_function

    def perform(self):
        """
        Perform the transformation on the dataset and return the result.

        Returns:
            The result of the transformation.
        """
        return self.transformation(self.dataset)

    def get_train_test_split(self, test_size=0.2, random_state=78563, path: str = "", save_split_sets=True, verbose=True):
        """
        Generates train/test indexes and splits the dataset into train and test subsets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int): The seed used by the random number generator. Defaults to 78563.
            path (str): The path to save the train/test indexes and splits. Defaults to "".
            save_splt_sets (bool): Indicates whether to save the train/test splits. Defaults to True.
            verbose (bool): Enable show messages of the process. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The train and test sets as numpy arrays.
        """

        train_idx_path = Path(f"{path}/train_{random_state}_{test_size}.npy")
        test_idx_path = Path(f"{path}/test_{random_state}_{test_size}.npy")

        if (len(path) == 0 or not train_idx_path.exists()):
            if verbose:
                print("Generating train/test indexes...")

            train_idx, test_idx = train_test_split(
                list(range(len(self.dataset))), test_size=test_size, random_state=random_state)

            if (len(path) != 0):
                if verbose:
                    print("Saving train/test indexes...")

                train_idx_path.parent.mkdir(exist_ok=True, parents=True)
                np.save(train_idx_path.absolute(), train_idx)
                np.save(test_idx_path.absolute(), test_idx)

        else:
            if verbose:
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
            train_data = self.__create_subset(train_subset, transformed_data.name, verbose=verbose)

            test_subset = Subset(transformed_data, test_idx if type(
                test_idx) == list else test_idx.tolist())            
            test_data = self.__create_subset(test_subset, transformed_data.name, verbose=verbose)


            if (len(path) != 0 and save_split_sets):
                if verbose:
                    print("Saving train/test splits...")

                train_subset_path.parent.mkdir(exist_ok=True, parents=True)
                np.save(train_subset_path.absolute(), train_data)
                np.save(test_subset_path.absolute(), test_data)
            
            train_data = np.array(train_data)
            test_data = np.array(test_data)
        else:
            if verbose:
                print("Loading train/test splits...")

            train_data = np.load(train_subset_path.absolute())
            test_data = np.load(test_subset_path.absolute())

        return train_data, test_data

    def __create_subset(self, subset, description, verbose = True):
        """
        Converts a Subset object into a numpy array.

        Args:
            subset (Subset): The Subset object to be converted.
            name (str): The name of the dataset.
            verbose (bool): Enable show messages of the process. Defaults to True.

        Returns:
            The converted subset.
        """
        result_data = []
        for data in tqdm(subset , desc=f"Generating train subset [{description}]", disable=not verbose):
            if torch.is_tensor(data):
                result_data.append(data.to(self.device).cpu().numpy())
            else:
                result_data.append(data)
        return result_data