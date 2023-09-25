from datasets import load_dataset, Audio
import youtube_dl
import ffmpeg
import static_ffmpeg
from pathlib import Path

from .dataset import Dataset

static_ffmpeg.add_paths()

class MusicCapsDataset(Dataset):
    """
    MusicCapsDataset
    =======

    Contains the class responsible for downloading the songs available in the 'google/MusicCaps' dataset.

    Args:
        format (str): The format of downloaded songs. Default: 'wav'.
    """

    def __init__(self, format="wav"):
        """Initializes a new MusicCapsDataset
        """
        super().__init__("MusicCaps")
        self.format = format

    def generate(self, sampling_rate = 48000, num_proc=1, batch_size=100, remove_failures = True):
        """
        Generates the 'MusicCaps' dataset. Download the songs if they are not found.

        Args:
            sampling_rate (int): Song sample rate. Default: 48000.
            num_proc (int): number of processes to be used during dataset generation. Default: 1.
            batch_size (int): Batch size per process. Default: 100.
            remove_failures (bool): Remove dataset download failures. Default: True.

        Returns:
            torch.Tensor: The MusicCaps dataset
        """
        super().generate()
        dataset = load_dataset('google/MusicCaps')

        def process(sample):
            output_path = f"{self.dataset_dir}/raw/{sample['ytid']}.{self.format}"
            online = Path(output_path).exists()

            if not online:
                temp_path = "undefined"
                try:
                    temp_path = self.__download_youtube_clip(sample['ytid'], sampling_rate)
                except:
                    pass
                finally:
                    online = Path(temp_path).exists()

                if online:
                    self.__clip(sample['ytid'], 
                                sample['start_s'], 
                                sample['end_s'] - sample['start_s'])
                else:
                    print(f"Failed to download '{sample['ytid']}'")

            sample['audio'] = output_path
            sample['online'] = online
            
            return sample

        dataset = dataset.map(
            process,
            num_proc=num_proc,
            writer_batch_size=batch_size,
            keep_in_memory=False
        )

        if remove_failures:
            dataset = dataset.filter(lambda sample : sample['online'])

        dataset = dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))

        dataset = dataset.with_format("torch")

        return dataset['train']

    def __download_youtube_clip(self, id : str, sampling_rate = 48000):
        """
        Download a YouTube video from an ID.

        Args:
            id (str): YouTube identifier.
            sampling_rate (int): Song sample rate. Default: 48000.
        
        Returns:
            str: The path of downloaded music.
        """
        path = f"{self.dataset_dir}/raw/{id}-temp.{self.format}"

        with youtube_dl.YoutubeDL({
                'quiet': True,
                'format': 'bestaudio/best',
                'outtmpl': path,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': f"{self.format}",
                    'preferredquality': str(sampling_rate)
                }]}) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={id}"])
        
        return path
    
    def __clip(self, id, start_time, duration, delete_temporary = True):
        """
        Cuts the song from a starting time and duration

        Args:
            id (str): YouTube identifier.
            start_time (float): Starting time.
            duration (float): Duration.
            delete_temporary (bool): Delete the original file. Default: True.
        
        Returns:
            str: The path of clipped music.
        """
        temp_path = f"{self.dataset_dir}/raw/{id}-temp.{self.format}"
        output_path = f"{self.dataset_dir}/raw/{id}.{self.format}"
        
        ffmpeg.input(temp_path).output(output_path, ss=start_time, t=duration).run(quiet=True)

        if delete_temporary:
            Path(temp_path).unlink(True)

        return output_path
    
