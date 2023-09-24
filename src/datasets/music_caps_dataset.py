from datasets import load_dataset, Audio
import youtube_dl
import ffmpeg
import static_ffmpeg
from pathlib import Path
from .dataset import Dataset

static_ffmpeg.add_paths()

class MusicCapsDataset(Dataset):

    def __init__(self, format="wav"):
        super().__init__("MusicCaps")
        self.format = format

    def generate(self, sampling_rate = 48000, num_proc=1, batch_size=100):
        super().generate()
        dataset = load_dataset('google/MusicCaps')

        def process(sample):
            output_path = f"{self.dataset_dir}/{id}.{self.format}"

            if not Path(output_path).exists():
                self.__download_youtube_clip(sample['ytid'], sampling_rate)

                self.__clip(sample['ytid'], 
                            sample['start_s'], 
                            sample['end_s'] - sample['start_s'])

            sample['audio'] = output_path
            sample['online'] = Path(output_path).exists()

        dataset.map(
            process,
            num_proc=num_proc,
            writer_batch_size=batch_size,
            keep_in_memory=False
        ).cast_column('audio', Audio(sampling_rate=sampling_rate))
        

    def __download_youtube_clip(self, id, sampling_rate = 48000):
        path = f"{self.dataset_dir}/{id}-temp.{self.format}"

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
        temp_path = f"{self.dataset_dir}/{id}-temp.{self.format}"
        output_path = f"{self.dataset_dir}/{id}.{self.format}"
        
        ffmpeg.input(temp_path).output(output_path, ss=start_time, t=duration).run(quiet=True)

        if delete_temporary:
            Path(temp_path).unlink(True)

        return output_path
    
