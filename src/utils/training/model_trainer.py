
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler
from audio_encoders_pytorch import MelE1d, TanhBottleneck

from src.datasets import MusicCapsDataset
from src.features import PreProcessor
from src.features.extractor import WaveformExtractor
from src.utils.data import TorchDataset, ModelCheckpoint
from src.utils.gpu import create_device

def create_model():
    """
    Create and initialize a DiffusionAE model.

    The model is constructed using a MelE1d encoder and various parameters
    defining its architecture, such as the number of channels, multipliers,
    and factors. The model is tailored for audio processing with a specific
    focus on handling Mel spectrogram data.

    Returns:
        A DiffusionAE model object ready for training or inference.
    """
    return DiffusionAE(
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

class ModelTrainer:
    def __init__(self, model_name, destination):
        """
        Initialize the ModelTrainer class.

        Args:
            model_name (str): Name of the model, used for organizing saved checkpoints.
            destination (str): File path where the final trained model will be saved.

        The constructor sets up the base path for model saving and loading operations.
        """
        self.model_name = model_name
        self.destination = destination
        self.base_path = f"models/{model_name}"

    def load_model(self): 
        """
        Load an existing model from the specified destination.

        This method attempts to load a saved model state from a given file path. 
        It ensures the model is loaded onto the appropriate computational device.

        Returns:
            A tuple of (model, device) if loading is successful; otherwise, (None, None).
        """
        if not os.path.exists(self.destination):
            return None, None
        device = create_device()
        model = create_model()
        model.load_state_dict(torch.load(self.destination, map_location=device))
        model = model.to(device)
        return model, device
    
    
    def test_data(self):
        musiccaps_generator = MusicCapsDataset(format="mp3", crop_length=5.5)
        dataset = musiccaps_generator.generate(num_proc=1)

        _, test = PreProcessor(dataset, lambda dataset: WaveformExtractor(dataset, column="audio", crop_length=2**18)).get_train_test_split(path=musiccaps_generator.get_processed_folder(), save_split_sets=False)
        return test


    def train(self):
        """
        Train the model using the MusicCaps dataset.

        This method handles the entire training process including dataset loading,
        preprocessing, model training, and saving checkpoints. The final trained
        model state is saved to the specified destination.

        The training process involves iterating over the dataset in batches, computing
        the loss, and updating the model parameters.
        """
        musiccaps_generator = MusicCapsDataset(format="mp3", crop_length=5.5)
        dataset = musiccaps_generator.generate(num_proc=1)

        train, test = PreProcessor(dataset, lambda dataset: WaveformExtractor(dataset, column="audio", crop_length=2**18)).get_train_test_split(path=musiccaps_generator.get_processed_folder(), save_split_sets=False)

        device = create_device()

        batch_size = 8
        transform = lambda x: x.unsqueeze(0)

        train_dataset = TorchDataset(train, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        autoencoder = create_model()

        autoencoder = autoencoder.to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        checkpoint_manager = ModelCheckpoint(self.model_name)
        start_epoch, start_batch_index, _ = checkpoint_manager.resume(autoencoder, optimizer)

        num_epochs = 10
        for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
            with tqdm(train_dataloader, unit="i", leave=False, desc="Batches") as tepoch:
                for i, batch in enumerate(tepoch):    
                    if epoch == start_epoch and i < start_batch_index:
                        continue

                    batch = batch.to(device)
                    optimizer.zero_grad()
                    loss = autoencoder(batch)
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % 10 == 0:
                        checkpoint_manager.save(autoencoder, optimizer, epoch, i, loss.item())
                        tepoch.set_description(f"Epoch {epoch} Batch {i+1}/{len(train_dataloader)} Loss: {loss.item():.4f} Saved")
                    else:
                        # Update progress
                        tepoch.set_description(f"Epoch {epoch} Batch {i+1}/{len(train_dataloader)} Loss: {loss.item():.4f}")
        

        # Save final model
        torch.save(autoencoder.state_dict(), self.destination)
        print(f"Training complete. The final model is saved to {self.destination}")


