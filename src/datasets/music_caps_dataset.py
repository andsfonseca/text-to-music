from pathlib import Path
import warnings
import logging

from datasets import load_dataset, Audio
import youtube_dl
import ffmpeg
import static_ffmpeg

from .dataset import Dataset

static_ffmpeg.add_paths()


class MusicCapsDataset(Dataset):
    """
    MusicCapsDataset
    =======

    Contains the class responsible for downloading the songs available in the 'google/MusicCaps' 
    dataset.

    Args:
        format (str): The format of downloaded songs. Default: 'wav'.
        sampling_rate (int): Song sample rate. Default: 48000.
        crop_length (int): The length of cropped songs. If crop_length less than 3, it will be 
        ignored. Default: 0.
    """

    def __init__(self, format="wav", sampling_rate=48000, crop_length=0):
        """Initializes a new MusicCapsDataset
        """
        super().__init__("MusicCaps")

        self.format = format
        self.sampling_rate = sampling_rate
        self.crop_length = crop_length

        self.logger = logging.getLogger("youtube_dl")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            f"{self.get_raw_folder()}/youtube_dl.log")
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        if 0 < crop_length and crop_length < 3:
            warnings.warn("Ignoring 'crop_length', the value is too small")

    def generate(self, num_proc=1, batch_size=100, remove_failures=True):
        """
        Generates the 'MusicCaps' dataset. Download the songs if they are not found.

        Args: 
            num_proc (int): number of processes to be used during dataset generation. Default: 1.
            batch_size (int): Batch size per process. Default: 100.
            remove_failures (bool): Remove dataset download failures. Default: True.

        Returns:
            Dataset: The MusicCaps dataset
        """
        super().generate()
        dataset = load_dataset('google/MusicCaps')

        dataset = dataset.map(
            self.__process,
            batched=True,
            num_proc=num_proc,
            writer_batch_size=batch_size,
            batch_size=batch_size,
            keep_in_memory=False
        )

        if remove_failures:
            dataset = dataset.filter(lambda sample: sample['online'])

        dataset = dataset.cast_column(
            'audio', Audio(sampling_rate=self.sampling_rate))

        dataset = dataset.with_format("torch")

        return dataset['train']

    def __process(self, samples):
        """
        Process the given samples and generate modified samples.
        
        Args:
            samples (dict): A dictionary containing the samples to be processed.
            
        Returns:
            dict: A dictionary containing the modified samples.
        """

        columns = list(samples.keys())
        keys_size = len(samples[columns[0]])

        result_samples = {key: [] for key in columns}
        result_samples['audio'] = []
        result_samples['online'] = []

        # for sample in samples:
        for i in range(keys_size):

            audio_output_path = f"{self.get_raw_folder()}/{samples['ytid'][i]}.{self.format}"
            online = Path(audio_output_path).exists()

            if not online:
                temp_path = "undefined"
                try:
                    # Try download the video
                    temp_path = self.__download_youtube_clip(
                        samples['ytid'][i], self.sampling_rate)
                except:
                    pass
                finally:
                    # Check if music is downloaded
                    online = Path(temp_path).exists()

                if online:
                    self.__clip(samples['ytid'][i],
                                samples['start_s'][i],
                                samples['end_s'][i] - samples['start_s'][i])
                else:
                    print(f"Failed to download '{samples['ytid'][i]}'")

            if online and self.crop_length >= 3:
                splited_paths = self.__split(
                    samples['ytid'][i], samples['end_s'][i] - samples['start_s'][i], self.crop_length)

                for j in range(len(splited_paths)):

                    # Copy current sample
                    split_sample = {}
                    for key in columns:
                        split_sample[key] = samples[key][i]

                    # Update aspect_list and caption
                    split_sample['aspect_list'] = split_sample['aspect_list'][:-1] + \
                        f", '{j+1} of {len(splited_paths)}'" + \
                        split_sample['aspect_list'][-1:]
                    split_sample['caption'] = split_sample['caption'] + \
                        f" {j+1} of {len(splited_paths)}."

                    # Add new columns
                    split_sample['audio'] = splited_paths[j]
                    split_sample['online'] = online

                    # Add to batch
                    for key in result_samples.keys():
                        result_samples[key].append(split_sample[key])
            else:

                # Copy current sample
                modified_sample = {}
                for key in columns:
                    modified_sample[key] = samples[key][i]

                # Add new columns
                modified_sample['audio'] = audio_output_path
                modified_sample['online'] = online

                for key in result_samples.keys():
                    result_samples[key].append(modified_sample[key])

        return result_samples

    def __download_youtube_clip(self, id: str, sampling_rate=48000):
        """
        Download a YouTube video from an ID.

        Args:
            id (str): YouTube identifier.
            sampling_rate (int): Song sample rate. Default: 48000.

        Returns:
            str: The path of downloaded music.
        """
        path = f"{self.get_raw_folder()}/{id}-temp.{self.format}"

        with youtube_dl.YoutubeDL({
                'quiet': True,
                'no_warnings': True,
                'verbose': False,
                'format': 'bestaudio/best',
                'outtmpl': path,
                'logger': self.logger,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': f"{self.format}",
                    'preferredquality': str(sampling_rate)
                }]}) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={id}"])

        return path

    def __clip(self, id: str, start_time: float, duration: float, delete_temporary=True):
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
        temp_path = f"{self.get_raw_folder()}/{id}-temp.{self.format}"
        output_path = f"{self.get_raw_folder()}/{id}.{self.format}"

        ffmpeg.input(temp_path).output(
            output_path, ss=start_time, t=duration).run(quiet=True)

        if delete_temporary:
            Path(temp_path).unlink(True)

        return output_path

    def __split(self, id: str, total_duration: float, split_time: float, preserve_last=True):
        """
        Splits an audio file into multiple segments based on the given split time.

        Args:
            id (str): The ID of the audio file.
            total_duration (float): The total duration of the audio file in seconds.
            split_time (float): The duration of each split segment in seconds.
            preserve_last (bool, optional): Whether to preserve the last segment if its duration is less than 
            the split time. Defaults to True.

        Returns:
            List[str]: A list of paths to the output split segments.
        """

        result_paths = []

        audio_path = f"{self.get_raw_folder()}/{id}.{self.format}"
        output_folder_path = f"{self.get_interim_folder()}/{str(split_time)}"

        Path(output_folder_path).mkdir(exist_ok=True, parents=True)

        times = [
            i * split_time for i in range(int(total_duration / split_time))]
        if preserve_last:
            times.append(total_duration - split_time)

        for i, start_time in enumerate(times):
            output_path = f"{output_folder_path}/{id}_{i+1}.{self.format}"
            result_paths.append(output_path)

            if (not Path(output_path).exists()):
                ffmpeg.input(audio_path).output(
                    output_path, ss=start_time, t=split_time).run(quiet=True, capture_stdout=True, capture_stderr=True)

        return result_paths
