import os
import json
from pathlib import Path
import time


class DDPDataExchanger():
    """
    DDPDataExchanger
    =======
    A class used to share data across multiple processes in Distributed Data Parallel (DDP) mode.
    """

    def __init__(self):
        """
        Initializes the DiffusionAE.
        """

        self.worker = True
        self.filepath = Path("ddp.share")

    def __enter__(self):
        """Executed when entering the context. Sets the worker flag based on the process rank.

        Returns:
            (DDPDataExchanger): Returns the DDPDataExchanger instance.
        """

        self.worker = not os.getenv("LOCAL_RANK", '0') == '0'

        return self

    def __exit__(self, type, value, traceback):
        """Executed when exiting the context. If the current process is not a worker, removes 
        the temporary file.

        Args:
            type (Any): Type of exception
            value (Any): Value of exception
            traceback (Any): StackTrace of exception
        """

        if not self.worker:
            self.filepath.unlink()

    def get_data(self):
        """
        Reads and returns the shared data from the file. If the file does not exist, waits for 1 
        second and tries again.

        Returns:
            (dict): The shared data
        """

        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                return json.load(f)
        else:
            time.sleep(1)
            return self.get_data()

    def set_data(self, data):
        """
        Writes the given data to the file. Only performed if the current process is not a worker.

        Args:
            data (dict): The data to be shared.
        """

        if not self.worker:
            with open(self.filepath.absolute(), "w") as f:
                f.write(json.dumps(data, indent=4))

    def is_worker(self):
        """
        Returns whether the current process is a worker.

        Returns:
            (bool): True if the current process is a worker, False otherwise.
        """
        return self.worker
