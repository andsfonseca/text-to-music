from pydub import AudioSegment
import numpy as np

class Audio:

    @staticmethod
    def save(audio, sample_rate, filename):
        """
        Save the generated audio to an MP3 file. This method normalizes the audio array and saves it as an MP3 file.

        Args:
            audio (ArrayLike): The audio data to be saved, as a numpy array.
            sample_rate (int): The sample rate of the audio.
            filename (str): The destination file for saving the audio.

        """

        # Normalize the audio to the range of 16-bit PCM
        audio = np.int16(audio / np.max(np.abs(audio)) * 32767)

        # Create an AudioSegment instance from the numpy array
        audio_segment = AudioSegment(
            audio.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio.dtype.itemsize, 
            channels=audio.shape[0]
        )

        # Export the audio segment to an MP3 file
        audio_segment.export(filename, format='mp3')
