import os
import torch
import logging
from datetime import datetime

class ModelCheckpoint:
    """
    ModelCheckpoint
    =======
    A class used to save and load checkpoints of a model during training.
    
    Args:
       filename (str): The name of the file where the checkpoint will be saved.

    """

    def __init__(self, model_name):
        """ Initialize the ModelCheckpoint
        """
        self.base_path = f"models/{model_name}"
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

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
        os.makedirs(save_path, exist_ok=True)
        filename = f"{save_path}/Checkpoint.ckpt"

        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'batch_index': batch_index,
            'loss': loss
        }
        torch.save(state, filename)
        print(f"Checkpoint salvo em {filename}")

    
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
            print(f"Retomando do checkpoint: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                start_batch_index = checkpoint.get('batch_index', 0)
                loss = checkpoint.get('loss')
                return start_epoch, start_batch_index, loss
            except Exception as e:
                print(f"Erro ao carregar o checkpoint: {e}")
                return 0, 0, None
        else:
            print("Nenhum checkpoint encontrado.")
            return 0, 0, None

    def __find_latest_checkpoint(self):
        all_checkpoints = [os.path.join(dirpath, f) 
                           for dirpath, dirnames, files in os.walk(self.base_path) 
                           for f in files if f.endswith('.ckpt')]
        if all_checkpoints:
            return max(all_checkpoints, key=os.path.getmtime)
        return None
