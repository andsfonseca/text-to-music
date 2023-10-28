import torch
from datasets.arrow_dataset import Dataset

from .extractor import Extractor


class WaveformExtractor(Extractor):
    """
    WaveformExtractor
    =======

    Extracts the Waveform information from a dataset. 

    Args:
        dataset (Dataset): The dataset that contains the audio.
        audio_column (str): The name of the audio column
        time_fixed (int): The fixed time duration of the waveform. When crop_length is not 0, 
        the waveform is cropped.
        crop_length (int): The length of the waveform. When time_fixed is not 0, the waveform 
        is fixed. crop_length has more priority than time_fixed.
    """

    def __init__(self, dataset: Dataset, column="audio", time_fixed=0, crop_length=0):
        """Initializes a new WaveformExtractor
        """
        super().__init__(dataset=dataset, column=column, name="Waveform")

        self.time_fixed = time_fixed
        self.crop_length = crop_length

    def __getitem__(self, idx):
        # Get the audio data
        audio_data = super().__getitem__(idx)

        # Audio Data Keys = 'path', 'array', and 'sampling_rate'
        waveform = audio_data['array']

        if self.crop_length != 0:
            waveform = self.__crop(waveform, self.crop_length)
        elif self.time_fixed != 0:
            sample_size = self.time_fixed * audio_data['sampling_rate']
            waveform = self.__crop(waveform, sample_size)

        return waveform

    def __crop(self, waveform, length):
        """
        Crop the waveform to a specified length.

        Parameters:
            waveform (torch.Tensor): The input waveform.
            length (int): The desired length of the waveform.

        Returns:
            torch.Tensor: The cropped waveform.
        """

        if waveform.size()[0] < length:
            pad = (0, length - waveform.size()[0])
            waveform = torch.nn.functional.pad(waveform, pad, value=0)
        else:
            waveform = waveform[:length]

        return waveform
