import torch
import argparse

# Import required modules
from src.datasets import MusicCapsDataset
from src.features.extractor import WaveformExtractor
from src.features import PreProcessor, DataModule
from src.models import DiffusionModel
from src.utils.gpu import create_device
from src.utils.audio import Audio
from src.utils.data import DDPDataExchanger

# Constants
BATCH_SIZE = 6
SAMPLING_RATE = 48000

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Generate audio using a Diffusion Model.')
    parser.add_argument('model_file', help='The name of the model file to use')
    parser.add_argument('prompt', help='The text prompt for audio generation')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps for audio generation (default: 100)')
    parser.add_argument('--num_audios', type=int, default=2, help='Number of audios to generate (default: 2)')
    args = parser.parse_args()

    # Load the model
    encoder = DiffusionModel.load_from_checkpoint(args.model_file)

    # Run model in evaluation mode
    encoder.eval()

    # Generate and save audio if not a worker
    device = create_device()
    model = encoder.model.to(device)

    print("Generating audios...")

    with torch.no_grad():
        for i in range(args.num_audios):
            # Generate audio sample with noise
            noise = torch.randn(1, 1, 2**18).to(device)
            
            sample = model.sample(
                noise,
                text=[args.prompt],  # Use the provided text prompt
                embedding_scale=15,
                num_steps=args.num_steps
            )

            # Save the generated audio
            Audio.save(sample.cpu().numpy(), sample_rate=SAMPLING_RATE, filename=f"audio_{i}.mp3")
            print(f"Audio {i} saved to audio_{i}.mp3")

