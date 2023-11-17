import datetime
from pathlib import Path

MODEL_ROOT_FOLDER = "models"


class BaseModel:
    """
    Base Model
    =======

    A base class for models that provides methods for managing model.

    Args:
        name (str): The name of the model. Defaults to "BaseModel".
    """

    def __init__(self, name="BaseModel"):
        """
        Initializes the BaseModel. 
        """
        self.model_dir = f"{MODEL_ROOT_FOLDER}/{name}"
        self.run_dir = ""

    def get_model_folder(self):
        """
        Returns the path to the model folder for saving checkpoints.

        Returns:
            str: The path to the raw folder.
        """
        Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        return self.model_dir

    def get_run_folder(self, force_new_folder=False):
        """
        Returns the path to the run folder. Creates a new run folder if `force_new_folder` is True or if there is no current run folder.

        Args:
            force_new_folder (bool): Whether to force the creation of a new run folder. Defaults to False.

        Returns:
            str: The path to the run folder.
        """
        if force_new_folder or self.run_dir == "":
            current_time = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
            self.run_dir = f"{self.get_model_folder()}/{current_time}"
            Path(self.run_dir).mkdir(exist_ok=True, parents=True)

        return self.run_dir
