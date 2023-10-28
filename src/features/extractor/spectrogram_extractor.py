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
        column (str): The name of the audio column
    """

    def __init__(self, dataset: Dataset, column):
        """Initializes a new SpectrogramExtractor
        """
        super().__init__(dataset, column=column, name="Spectrogram")

    def __getitem__(self, idx):
        waveform = super().__getitem__(idx)

        # Get the audio data
        audio_data = self.dataset[idx][self.column]

        spectrogram = torchaudio.transforms.MelSpectrogram(
            audio_data['sampling_rate'])(waveform)
        return spectrogram
