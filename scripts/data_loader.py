from pathlib import Path
import pandas as pd

class DataLoader:
    """
    A class to handle loading of a dataset from a CSV file.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DataLoader with the path to the dataset file.

        Args:
            file_path (str): The path to the CSV file containing the dataset.
        """
        self.file_path = Path(file_path)
        self.dataset = None

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified CSV file.

        Returns:
            pd.DataFrame: The loaded dataset as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.dataset = pd.read_csv(self.file_path)
        print(f"Dataset loaded with shape: {self.dataset.shape}")
        return self.dataset
