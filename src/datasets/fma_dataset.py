from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
import urllib.request
from zipfile import ZipFile, error as ZipFileException

from datasets import Audio, Dataset as HFDataset
import pandas as pd

from .dataset import Dataset
from ..utils import GenericProgressBar

FMA_METADATA_URL = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'

FMA_METADATA_IGNORED_FILES = [
    "fma_metadata/checksums",
    "fma_metadata/echonest.csv",
    "fma_metadata/features.csv",
    "fma_metadata/not_found.pickle",
    "fma_metadata/raw_albums.csv",
    "fma_metadata/raw_artists.csv",
    "fma_metadata/raw_echonest.csv",
    "fma_metadata/raw_genres.csv",
    "fma_metadata/raw_tracks.csv",
    "fma_metadata/README.txt"
]


class FMADasasetRepository(str, Enum):
    SMALL = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
    MEDIUM = 'https://os.unil.cloud.switch.ch/fma/fma_medium.zip'
    LARGE = 'https://os.unil.cloud.switch.ch/fma/fma_large.zip'
    FULL = 'https://os.unil.cloud.switch.ch/fma/fma_full.zip'


class FMADataset(Dataset):
    """
    FMADataset
    =======

    Contains the class responsible for downloading the songs available in the 'https://github.com/mdeff/fma' repository.

    Args:
        dataset_size (FMADasasetRepository): The subset of musics. Default: 'SMALL'.
    """

    def __init__(self, dataset_size=FMADasasetRepository.SMALL):
        """Initializes a new FMADataset.
        """
        super().__init__("FMA")
        self.dataset_size = dataset_size

    def generate(self, num_proc=1, batch_size=100, remove_failures=True):
        """
        Generates the 'FMA' dataset. Download the songs if they are not found and download all metadata.

        Args:
            num_proc (int): number of processes to be used during dataset generation. Default: 1.
            batch_size (int): Batch size per process. Default: 100.
            remove_failures (bool): Remove dataset download failures. Default: True.

        Returns:
            Dataset: The FMA dataset

        """
        super().generate()
        self.__extract(self.__download_metadata(), FMA_METADATA_IGNORED_FILES)
        self.__extract(self.__download_dataset())

        df = pd.read_csv(f"{self.dataset_dir}/raw/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
        df.columns = [f'{i}/{j}' for i, j in df.columns]

        dataset = HFDataset.from_pandas(df)

        audio_paths = self.__get_all_mp3(f"{self.dataset_dir}/raw/fma_{self.dataset_size.name.lower()}")
        def process(sample):
            sample['audio'] = ""
            sample['online'] = str(sample['track_id']) in audio_paths
            if sample['online']:
                sample['audio'] = str(audio_paths[str(sample['track_id'])].absolute())
            
            return sample
        
        dataset = dataset.map(
            process,
            num_proc=num_proc,
            writer_batch_size=batch_size,
            keep_in_memory=False
        )

        if remove_failures:
            dataset = dataset.filter(lambda sample : sample['online'])

        dataset = dataset.cast_column('audio', Audio(sampling_rate=44100))
        dataset = dataset.with_format("torch")

        return dataset

    def __download_dataset(self):
        """
        Download the 'FMA' dataset with the specific subset.

        Returns:
            str: A compressed file with the specific subset.

        """
        type = self.dataset_size.name
        url = self.dataset_size.value

        output_path = f"{self.dataset_dir}/raw/{type}.zip"

        path = Path(output_path)

        if not path.exists():
            path.parent.mkdir(exist_ok=True, parents=True)
            with GenericProgressBar(unit='B', unit_scale=True,
                                     miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(
                    url, filename=output_path, reporthook=t.update_to)

        return output_path

    def __download_metadata(self):
        """
        Download the 'FMA' metada.

        Returns:
            str: A compressed file with the all metadata.

        """

        output_path = f"{self.dataset_dir}/raw/metadata.zip"

        path = Path(output_path)

        if not path.exists():
            path.parent.mkdir(exist_ok=True, parents=True)
            with GenericProgressBar(unit='B', unit_scale=True,
                                     miniters=1, desc=FMA_METADATA_URL.split('/')[-1]) as t:
                urllib.request.urlretrieve(
                    FMA_METADATA_URL, filename=output_path, reporthook=t.update_to)

        return output_path

    def __extract(self, path, ignored_files = []):
        """
        Extract a file.

        Args:
            path (str): A path of the compressed file.
            ignored_files (list): A list of file to ignore when extracting. Default: [].

        """
        with ZipFile(path, 'r') as zf:

            directory_path = Path(path).parent
            total_size = sum((file.file_size for file in zf.infolist()))

            with GenericProgressBar(total=total_size, unit='B', unit_scale=True, miniters=1, desc='Extracting') as pbar:
                for member in zf.infolist():
                    target_file_path = Path(directory_path) / member.filename
                    if not member.filename in ignored_files and not target_file_path.exists():
                        try:
                            zf.extract(member, path=directory_path)
                            pbar.update(member.file_size)
                        except ZipFileException as e:
                            pass
                    else:
                        pbar.update(member.file_size)

    def __get_all_mp3(self, path):
        """
        Find all mp3 files inside a folder and your subfolders.

        Args:
            path (str): folder path.

        Returns:
            dict: A dictionary with all track_id as key and the path to the music as values.

        """
        folder = Path(path)
        pattern = "*.mp3"

        runs = {}

        for path in folder.rglob('*'):
            if path.is_file() and fnmatch(path.name, pattern):
                runs[path.stem] = path.absolute()
        return runs
