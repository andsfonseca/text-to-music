import os
import torch

class ModelCheckpoint:
    def __init__(self, filename):
        self.filename = filename

    def save(self, model, optimizer, epoch, batch_index, loss):
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'batch_index': batch_index,
            'loss': loss
        }
        torch.save(state, self.filename)
        print(f"Checkpoint saved to {self.filename}")

    def resume(self, model, optimizer):
        if os.path.isfile(self.filename):
            print(f"Resuming from checkpoint: {self.filename}")
            checkpoint = torch.load(self.filename)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_batch_index = checkpoint.get('batch_index', 0)
            loss = checkpoint.get('loss')
            return start_epoch, start_batch_index, loss
        else:
            print(f"No checkpoint found at {self.filename}")
            return 0, 0, None