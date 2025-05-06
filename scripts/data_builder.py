import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scripts.text_preprocessing import TextPreprocessor
import os
import json

class DatasetBuilder:
    """
    Class that handles the construction of a structured dataset for NLP classification tasks,
    including text preprocessing and label encoding.
    """

    def __init__(self, df: pd.DataFrame, text_column: str = "tweet_text", label_column: str = "cyberbullying_type"):
        """
        Initializes the DatasetBuilder with the input DataFrame and relevant column names.

        Args:
            df (pd.DataFrame): The input dataset containing text and labels.
            text_column (str): Name of the column containing the raw text.
            label_column (str): Name of the column containing the class labels.
        """
        if df.empty:
            raise ValueError("The input DataFrame is empty.")
        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column
        self.preprocessor = TextPreprocessor()
        self.label_encoder = LabelEncoder()
        
    def _check_column_exists(self, column_name: str):
        """Helper method to check if a column exists in the DataFrame."""
        if column_name not in self.df.columns:
            raise KeyError(f"Column '{column_name}' not found in the dataset.")

    def add_soft_text(self, target_column: str = "tweet_soft"):
        """
        Adds a new column to the DataFrame with softly preprocessed text 
        (e.g., light cleaning, minimal transformation).

        Args:
            target_column (str): Name of the new column to store the processed text.
        """
        self._check_column_exists(self.text_column)
        self.df[target_column] = self.df[self.text_column].astype(str).apply(self.preprocessor.clean_text_soft)

    def add_full_text(self, target_column: str = "tweet_full"):
        """
        Adds a new column to the DataFrame with fully preprocessed text 
        (e.g., extensive cleaning, normalization, removal of noise).

        Args:
            target_column (str): Name of the new column to store the processed text.
        """
        self._check_column_exists(self.text_column)
        self.df[target_column] = self.df[self.text_column].astype(str).apply(self.preprocessor.clean_text_full)

    def add_binary_label(self, target_column: str = "is_cyberbullying"):
        """
        Adds a binary label column where 'not_cyberbullying' is mapped to 0, 
        and any other class is mapped to 1.

        Args:
            target_column (str): Name of the new column to store binary labels.
        """
        self._check_column_exists(self.text_column)
        self.df[target_column] = self.df[self.label_column].apply(lambda x: 0 if x == "not_cyberbullying" else 1)

    def add_multiclass_label(self, target_column: str = "cyberbullying_label"):
        """
        Encodes the categorical label column into numeric form using LabelEncoder.

        Args:
            target_column (str): Name of the new column to store encoded labels.
        """
        self._check_column_exists(self.text_column)
        self.df[target_column] = self.label_encoder.fit_transform(self.df[self.label_column])
        
    def save_label_mapping(self, output_path: str):
        """
        Saves the mapping of original string labels to numeric values as a JSON file.

        Args:
            output_path (str): File path where the label mapping will be saved.
        """
        # Create a dictionary mapping label strings to integers
        try:
            # Check if label encoder has been fitted
            if not hasattr(self.label_encoder, 'classes_'):
                raise AttributeError("Label encoder has not been fitted yet. Run add_multiclass_label() first.")
            
            label_map = {
                str(label): int(idx)
                for label, idx in zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_)
                )
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(label_map, f, indent=2)
            print(f"Label mapping saved to {output_path}.")
        
        except (OSError, IOError) as e:
            print(f"Error saving label mapping to {output_path}: {e}")
        except AttributeError as e:
            print(e)

    def build(self, include_soft=True, include_full=True, binary=True, multiclass=True):
        """
        Executes the dataset construction pipeline based on provided flags.

        Args:
            include_soft (bool): Whether to add softly preprocessed text.
            include_full (bool): Whether to add fully preprocessed text.
            binary (bool): Whether to add a binary classification label.
            multiclass (bool): Whether to add a multiclass encoded label.

        Returns:
            pd.DataFrame: The augmented dataset with added features and labels.
        """
        if include_soft:
            self.add_soft_text()
        if include_full:
            self.add_full_text()
        if binary:
            self.add_binary_label()
        if multiclass:
            self.add_multiclass_label()
        return self.df
