import torch
from datasets.arrow_dataset import Dataset

from .extractor import Extractor


class WaveformExtractor(torch.utils.data.Dataset, Extractor):
    """
    WaveformExtractor
    =======

    Extracts the Waveform information from a dataset.

    Args:
        dataset (Dataset): The dataset that contains the audio.
        audio_column (str): The name of the audio column
        time_fixed (int): The fixed length of the waveform.
        name (str): The name of the extractor
    """

    def __init__(self, dataset: Dataset, audio_column="audio", time_fixed=0, name="Waveform", ):
        """Initializes a new WaveformExtractor
        """
        Extractor.__init__(self, name)

        self.dataset = dataset
        self.audio_column = audio_column
        self.time_fixed = time_fixed

        # Define the device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the audio data
        audio_data = self.dataset[idx][self.audio_column]

        # Audio Data Keys = 'path', 'array', and 'sampling_rate'
        waveform = audio_data['array']

        if self.time_fixed != 0:
            sample_size = self.time_fixed * audio_data['sampling_rate']

            if waveform.size()[0] < sample_size:
                pad = (0, sample_size - waveform.size()[0])
                waveform = torch.nn.functional.pad(waveform, pad, value=0)
            else:
                waveform = waveform[:sample_size]

        return waveform
