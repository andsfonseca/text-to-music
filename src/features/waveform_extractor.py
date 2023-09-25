import torch
from datasets.arrow_dataset import Dataset

class WaveformExtractor(torch.utils.data.Dataset):
    """
    WaveformExtractor
    =======

    Extracts the Waveform information from a dataset.

    Args:
        dataset (Dataset): The dataset that contains the audio.
        audio_column (str): The name of the audio column
    """
    
    def __init__(self, dataset : Dataset, audio_column):
        """Initializes a new WaveformExtractor
        """

        self.dataset = dataset
        self.audio_column = audio_column

        # Define the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the audio data
        audio_data = self.dataset[idx][self.audio_column]

        # Audio Data Keys = 'path', 'array', and 'sampling_rate'
        waveform = audio_data['array']
        sample_rate = audio_data['sampling_rate']

        return waveform, sample_rate