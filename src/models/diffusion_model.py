import torch.optim as optim
import pytorch_lightning as pl

from audio_diffusion_pytorch import DiffusionModel as DiffusionModelBase
from audio_diffusion_pytorch import UNetV0, VDiffusion, VSampler

from .base_model import BaseModel


class DiffusionModel(BaseModel, pl.LightningModule):
    """
    DiffusionModel
    =======
    The Diffusion Model.
    """

    def __init__(self):
        """
        Initializes the DiffusionModel. Sets up the Diffusion Model.
        """

        BaseModel.__init__(self, "DiffusionModel")
        pl.LightningModule.__init__(self)

        self.model = DiffusionModelBase(
            net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
            in_channels=1, # U-Net: number of input/output (audio) channels
            channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
            factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
            items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
            attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
            attention_heads=8, # U-Net: number of attention heads per attention item
            attention_features=64, # U-Net: number of attention features per attention item
            diffusion_t=VDiffusion, # The diffusion method used
            sampler_t=VSampler, # The diffusion sampler used
            use_text_conditioning=True, # U-Net: enables text conditioning (default T5-base)
            use_embedding_cfg=True, # U-Net: enables classifier free guidance
            embedding_max_length=64, # U-Net: text embedding maximum length (default for T5-base)
            embedding_features=768, # U-Net: text mbedding features (default for T5-base)
            cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.model(batch[0], text=batch[1], embedding_mask_proba=0.1)
        self.log('train_loss', loss)
        return loss

    def configure_logger(self):
        """
        Configures the logger for training.

        Returns:
            pl.loggers.TensorBoardLogger: The logger for training.
        """
        return pl.loggers.TensorBoardLogger(self.log_dir)
