from pydub import AudioSegment
import numpy as np

class Audio:

    @staticmethod
    def save(audios, sample_rate, folder_path):
        """
        Save the generated audio to an MP3 file. This method normalizes the audio array and saves it as an MP3 file.

        Args:
            audios (ArrayLike): The audios data to be saved, as a numpy array.
            sample_rate (int): The sample rate of the audios.
            folder_path (str): The destination folder for saving the audios.

        """

        # Normalize the audio to the range of 16-bit PCM
        audios = np.int16(audios / np.max(np.abs(audios)) * 32767)

        for i in range(audios.shape[0]):
            audio = audios[i]

            # Create an AudioSegment instance from the numpy array
            audio_segment = AudioSegment(
                audio.tobytes(), 
                frame_rate=sample_rate,
                sample_width=audio.dtype.itemsize, 
                channels=audio.shape[0]
            )

            # Export the audio segment to an MP3 file
            audio_segment.export(f"{folder_path}/audio_{i}", format='mp3')
