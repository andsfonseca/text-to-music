import matplotlib.pyplot as plt
import librosa
import numpy as np
from numpy.typing import ArrayLike

class AudioViewer():
    """
    AudioViewer
    =======

    Stores functions for viewing audio
    """

    @staticmethod
    def get_waveform(waveform: ArrayLike, sample_rate: float):
        """
        Generates a visualization for each channel of waveform

        Args:
            waveform (ArrayLike): Waveform Information.
            sample_rate (float): Sampling rate.
        """
        
        num_channels = waveform.shape[0]

        fig = plt.figure(figsize=(10, 2 * num_channels),
                         constrained_layout=False)

        grid = fig.add_gridspec(num_channels, 1, wspace=0, hspace=0)
        axes = grid.subplots(sharex=True, sharey=True)

        if num_channels == 1:
            axes = [axes]

        # Fix error prop_cycle MatplotLib
        prop_cycle = iter(plt.rcParams['axes.prop_cycle'])

        for i in range(num_channels):
            color = next(prop_cycle)['color']
            librosa.display.waveshow(
                waveform[i], sr=sample_rate, ax=axes[i], color='#3d3d3d')
            axes[i].label_outer()
            axes[i].set_facecolor(color)
            axes[i].set(xlabel=None)
            axes[i].set(xlim=[0, librosa.get_duration(
                y=waveform[i], sr=sample_rate)])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_spectogram(spectrogram: ArrayLike, sample_rate: float):
        """
        Generates a visualization for the spectrogram

        Args:
            spectrogram (ArrayLike): Spectrogram Information.
            sample_rate (float): Sampling rate.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(
            spectrogram, ref=np.max), sr=sample_rate, y_axis='hz', x_axis='time', hop_length=200)
        plt.title('Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
