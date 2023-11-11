import torch
from pydub import AudioSegment
import numpy as np

class AudioGenerator:
    def __init__(self, model, device):
        """
        Initialize the AudioGenerator class.

        Args:
            model (torch.nn.Module): The trained model used for audio generation.
            device (torch.device): The computational device (CPU or GPU) the model is on.

        This constructor stores the model and device, which are used later for generating audio.
        """
        self.model = model
        self.device = device

    def save_audio(self, audio, sample_rate, filename):
        """
        Save the generated audio to an MP3 file.

        Args:
            audio (torch.Tensor): The audio data to be saved, as a PyTorch tensor.
            sample_rate (int): The sample rate of the audio.
            filename (str): The destination file name for saving the audio.

        This method converts the audio tensor to a numpy array, normalizes it, and saves it as an MP3 file.
        """
        # Convert the audio from a PyTorch tensor to a numpy array
        audio_np = audio.cpu().numpy()

        # Normalize the audio to the range of 16-bit PCM
        audio_np = np.int16(audio_np / np.max(np.abs(audio_np)) * 32767)

        # Create an AudioSegment instance from the numpy array
        audio_segment = AudioSegment(
            audio_np.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio_np.dtype.itemsize, 
            channels=1
        )

        # Export the audio segment to an MP3 file
        audio_segment.export(filename, format='mp3')

    def generate_music(self, audio, num_steps=1000):
        """
        Generate music using the model.

        Args:
            audio (torch.Tensor): The audio data to be transformed, represented as a PyTorch tensor.
            num_steps (int, optional): The number of steps for the generation process. Defaults to 1000.

        Returns:
            torch.Tensor: The generated audio as a PyTorch tensor.

        This method generates audio by feeding another audio through the model and decoding the output.
        """
        audio = audio.to(next(self.model.parameters()).device)
        self.model.eval() 

        with torch.no_grad():
            latent = self.model.encode(audio)
            generated_audio = self.model.decode(latent, num_steps=num_steps)

        return generated_audio.cpu()
