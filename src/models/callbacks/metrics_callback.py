
import copy
from pytorch_lightning.callbacks import Callback

class MetricsCallback(Callback):
    """A PyTorch Lightning callback that logs the metrics at the end of each train epoch.

    This callback logs the metrics computed by the trainer at the end of each train epoch and stores them in a
    list. The logged metrics can be accessed through the `metrics` attribute of the callback.

    Args:
        metrics (list[dict[str, float]]): A list of dictionaries containing the logged metrics at the end of each
            train epoch. Each dictionary contains the metric names as keys and their values as values.

    """
    def __init__(self):
        "Initializes MetricsCallback"
        super().__init__()
        self.metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Log the metrics at the end of each train epoch.

        This method is called by the trainer at the end of each train epoch. It logs the metrics computed by
        the trainer and stores them in a list.

        Args:
            trainer (Trainer): The trainer instance.
            pl_module (LightningModule): The LightningModule instance being trained.
            
        """
        self.metrics.append(copy.deepcopy(trainer.callback_metrics))