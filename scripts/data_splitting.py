from sklearn.model_selection import train_test_split
import pandas as pd


class DataSplitter:
    """
    Class to handle data splitting into train, validation, and test sets.
    """
    def __init__(self, df: pd.DataFrame, label_column: str = 'cyberbullying_type', random_state: int = 42):
        self.df = df
        self.label_column = label_column
        self.random_state = random_state

    def split(self):
        """
        Splits the dataset into 90% dev (train + val) and 10% test,
        then splits dev into 81% train and 19% validation.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # First split: 90% dev, 10% test
        dev_df, test_df = train_test_split(
            self.df, test_size=0.10, stratify=self.df[self.label_column], random_state=self.random_state
        )

        # Second split: dev into 81% train, 19% val
        train_df, val_df = train_test_split(
            dev_df, test_size=0.19 / 0.90, stratify=dev_df[self.label_column], random_state=self.random_state
        )

        print(f"Train set: {train_df.shape[0]} samples")
        print(f"Validation set: {val_df.shape[0]} samples")
        print(f"Test set: {test_df.shape[0]} samples")

        return train_df, val_df, test_df