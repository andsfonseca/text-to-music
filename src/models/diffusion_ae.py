import torch.optim as optim
import pytorch_lightning as pl

from audio_diffusion_pytorch import DiffusionAE as DiffusionAEBase
from audio_diffusion_pytorch import UNetV0, VDiffusion, VSampler
from audio_encoders_pytorch import MelE1d, TanhBottleneck

from .base_model import BaseModel


class DiffusionAE(BaseModel, pl.LightningModule):
    """
    DiffusionAE
    =======
    The Diffusion Autoencoder model.
    """

    def __init__(self):
        """
        Initializes the DiffusionAE. Sets up the Diffusion Autoencoder model.
        """

        BaseModel.__init__(self, "DiffusionAE")
        pl.LightningModule.__init__(self)

        self.model = DiffusionAEBase(
            encoder=MelE1d(
                in_channels=1,
                channels=512,
                multipliers=[1, 1],
                factors=[2],
                num_blocks=[12],
                out_channels=32,
                mel_channels=80,
                mel_sample_rate=48000,
                mel_normalize_log=True,
                bottleneck=TanhBottleneck(),
            ),
            inject_depth=6,
            net_t=UNetV0,
            in_channels=1,
            channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
            factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
            items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('train_loss', loss)
        return loss

    def configure_logger(self):
        """
        Configures the logger for training.

        Returns:
            pl.loggers.TensorBoardLogger: The logger for training.
        """
        return pl.loggers.TensorBoardLogger(self.log_dir)
