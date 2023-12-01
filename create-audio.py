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

    # Initialize shared data exchange for distributed data parallelism
    with DDPDataExchanger() as shared_dict:
        # Only on RANK 0
        if not shared_dict.is_worker():
            run_folder = DiffusionModel().get_run_folder()
            shared_dict.set_data({"run_folder": run_folder})

        run_folder = shared_dict.get_data()["run_folder"]

        # Dataset setup
        dataset_generator = MusicCapsDataset(format="mp3", crop_length=20)
        dataset = dataset_generator.generate()

        # Preprocess the dataset
        preprocessor = PreProcessor(dataset=dataset,
                                    transformation_function=lambda dataset: WaveformExtractor(dataset,
                                                                                              column="audio",
                                                                                              crop_length=2**19))
        musiccaps_data = DataModule(preprocessors=[preprocessor],
                                    path=dataset_generator.get_processed_folder(),
                                    batch_size=BATCH_SIZE,
                                    transformations=lambda x: x.unsqueeze(0))
        musiccaps_data.setup()

        # Load the model
        encoder = DiffusionModel.load_from_checkpoint(args.model_file)
        encoder.set_run_folder(run_folder)

        # Run model in evaluation mode
        encoder.eval()

        # Generate and save audio if not a worker
        if not shared_dict.is_worker():
            device = create_device()
            model = encoder.model.to(device)

            print("Generating audios...")

            with torch.no_grad():
                for i, batch in enumerate(musiccaps_data.test_dataloader()):
                    # Generate audio sample
                    input_data = batch

                    # Move data to the same device as the model
                    input_data = input_data.to(device)
                    sample = model.sample(
                        input_data,
                        text=[args.prompt],  # Use the provided text prompt
                        embedding_scale=15,
                        num_steps=args.num_steps
                    )

                    # Save the generated audio
                    Audio.save(sample.cpu().numpy(), sample_rate=SAMPLING_RATE, filename=f"{run_folder}/audio_{i}.mp3")
                    print(f"Audio {i} saved to {run_folder}/audio_{i}.mp3")

                    # Limit the number of generated audios
                    if i == args.num_audios - 1:
                        break
