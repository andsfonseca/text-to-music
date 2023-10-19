class Extractor():
    """
    Extractor
    =======

    A abstract class that extracts the information from a dataset

    Args:
        name (str): The name of the extractor. Default: "EmptyExtractor".
    """

    def __init__(self, name="EmptyExtractor"):
        """
        Initializes a new Extractor
        """
        self.name = name
