import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from scipy.io.wavfile import write


from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler
from audio_encoders_pytorch import MelE1d, TanhBottleneck

from src.datasets import MusicCapsDataset
from src.features import PreProcessor
from src.features.extractor import WaveformExtractor
from src.utils.data import TorchDataset, ModelCheckpoint

def create_device():
    # Device selection priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    # mps is breaking during training    
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        return torch.device('cpu')

def create_model():
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

def train(final_model_path):
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

    checkpoint_path = "autoencoder_checkpoint.pth"
    checkpoint_manager = ModelCheckpoint("autoencoder_checkpoint.pth")
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
                    checkpoint_manager.save(autoencoder, optimizer, epoch, i, loss.item(), checkpoint_path)
                    tepoch.set_description(f"Epoch {epoch} Batch {i+1}/{len(train_dataloader)} Loss: {loss.item():.4f} Saved")
                else:
                    # Update progress
                    tepoch.set_description(f"Epoch {epoch} Batch {i+1}/{len(train_dataloader)} Loss: {loss.item():.4f}")
    

    # Save final model
    final_model_path = "final_autoencoder_model.pth"
    torch.save(autoencoder.state_dict(), final_model_path)
    print(f"Training complete. The final model is saved to {final_model_path}")

def save_audio(audio, sample_rate, filename):
    write(filename, sample_rate, audio.cpu().numpy())

def generate_music(model, device, num_samples=1, num_steps=1000):
    audio = torch.randn(num_samples, 1, 2**18, device=device)

    audio = audio.to(next(model.parameters()).device)

    model.eval() 

    with torch.no_grad():
        latent = model.encode(audio)
        generated_audio = model.decode(latent, num_steps=num_steps)

    return generated_audio.cpu()

if __name__ == "__main__":

    final_model_path = "final_autoencoder_model.pth"

    if not os.path.isfile(final_model_path):
        train(final_model_path)

    device = create_device()
    model = create_model()
    model.load_state_dict(torch.load(final_model_path, map_location=device))
    model = model.to(device)
    
    print("Generating music...")
    generated_audio = generate_music(model, device)

    # Save generated audio to a file
    output_filename = "generated_music.wav"
    sample_rate = 48000  
    save_audio(generated_audio.squeeze(), sample_rate, output_filename)
    print(f"Generated music saved to {output_filename}")    


    

