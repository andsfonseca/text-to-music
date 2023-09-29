from tqdm import tqdm

class GenericProgressBar(tqdm):
    """
    GenericProgressBar
    =======

    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Args:
            num_proc (int): number of processes to be used during dataset generation. Default: 1.
            batch_size (int): Batch size per process. Default: 100.
            remove_failures (bool): Remove dataset download failures. Default: True.

        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)