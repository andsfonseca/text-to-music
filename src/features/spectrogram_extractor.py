from datasets.arrow_dataset import Dataset
import torchaudio

from .waveform_extractor import WaveformExtractor

class SpectrogramExtractor(WaveformExtractor):
    """
    SpectrogramExtractor
    =======

    Extracts the Spectrogram information from a dataset.

    Args:
        dataset (Dataset): The dataset that contains the audio.
        audio_column (str): The name of the audio column
    """

    def __init__(self, dataset: Dataset, audio_column):
        """Initializes a new SpectrogramExtractor
        """
        super().__init__(dataset, audio_column)

    def __getitem__(self, idx):
        waveform, sample_rate = super().__getitem__(idx)
        spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)

        return spectrogram, sample_rate