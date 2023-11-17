import os
from datetime import datetime
from pathlib import Path

import torch


class ModelCheckpoint:
    """
    ModelCheckpoint
    =======
    A class used to save and load checkpoints of a model during training.

    Args:
        model_name (str): The name of the model.

    """

    def __init__(self, model_name):
        """ Initialize the ModelCheckpoint
        """
        self.base_path = f"models/{model_name}"
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, epoch, batch_index, loss):
        """
        Saves the current state of the model, optimizer, epoch, batch index, and loss to a checkpoint file.
        Args:
            model (torch.nn.Module): The model to be saved.
            optimizer (torch.optim.Optimizer): The optimizer to be saved.
            epoch (int): The current epoch.
            batch_index (int): The current batch index.
            loss (float): The current loss.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f"{self.base_path}/{timestamp}"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        filename = f"{save_path}/Checkpoint.ckpt"

        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'batch_index': batch_index,
            'loss': loss
        }
        torch.save(state, filename)
        print(f"Checkpoint saved in {filename}")

    def resume(self, model, optimizer):
        """
        Loads the state of the model and optimizer from a checkpoint file and returns the epoch, batch index, and loss.
        Args:
            model (torch.nn.Module): The model to be loaded.
            optimizer (torch.optim.Optimizer): The optimizer to be loaded.
        Returns:
            start_epoch (int): The epoch from the checkpoint.
            start_batch_index (int): The batch index from the checkpoint.
            loss (float): The loss from the checkpoint.
        """
        latest_checkpoint = self.__find_latest_checkpoint()
        if latest_checkpoint:
            print(f"Resuming checkpoint: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                start_batch_index = checkpoint.get('batch_index', 0)
                loss = checkpoint.get('loss')
                return start_epoch, start_batch_index, loss
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                return 0, 0, None
        else:
            print("No checkpoints found.")
            return 0, 0, None

    def __find_latest_checkpoint(self):
        """
        Finds the latest checkpoint file in the base directory.

        Returns:
            The path to the latest checkpoint file, or None if no checkpoint files are found.
        """

        all_checkpoints = [os.path.join(dirpath, f)
                           for dirpath, dirnames, files in os.walk(self.base_path)
                           for f in files if f.endswith('.ckpt')]
        if all_checkpoints:
            return max(all_checkpoints, key=os.path.getmtime)
        return None
