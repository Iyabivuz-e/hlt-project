import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSaver:
    """
    Class for managing the saving of datasets, including full datasets
    and train/test splits.
    """
    def __init__(self):
        """
        Initializes the DataSaver class.
        """
        pass

    def save_dataframe(self, df, output_path, create_dir=True):
        """
        Saves a DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            output_path (str): Path where the CSV will be saved.
            create_dir (bool, optional): Whether to create the directory if it does not exist. Defaults to True.

        Returns:
            None
        """
        try:
            if create_dir:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"Dataset saved to {output_path}.")
        except (OSError, IOError) as e:
            print(f"Error saving the DataFrame to {output_path}: {e}")

    def save_train_test_split(self, df, output_dir, test_size=0.1, stratify_col='target', random_state=42):
        """
        Splits a dataset into train and test sets and saves them as CSV files.

        Args:
            df (pd.DataFrame): DataFrame to split.
            output_dir (str): Directory where train and test CSVs will be saved.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.1.
            stratify_col (str, optional): Column name to use for stratified splitting. Defaults to 'target'.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            None
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            if stratify_col and stratify_col not in df.columns:
                raise ValueError(f"Stratify column '{stratify_col}' not found in DataFrame.")

            df_train, df_test = train_test_split(
                df,
                test_size=test_size,
                stratify=df[stratify_col] if stratify_col else None,
                random_state=random_state
            )

            train_path = os.path.join(output_dir, 'train.csv')
            test_path = os.path.join(output_dir, 'test.csv')

            df_train.to_csv(train_path, index=False)
            df_test.to_csv(test_path, index=False)

            print(f"Train/Test split saved to {output_dir}. Sizes: train={df_train.shape}, test={df_test.shape}")

        except ValueError as ve:
            print(f"Value error during train/test split: {ve}")
        except (OSError, IOError) as e:
            print(f"Error saving the train/test split to {output_dir}: {e}")

    def save_full_dataset(self, df, output_path):
        """
        Saves the full dataset, typically including cleaned text, target labels,
        and possibly one-hot encoded columns.

        Args:
            df (pd.DataFrame): DataFrame to save.
            output_path (str): Path where the full dataset CSV will be saved.

        Returns:
            None
        """
        self.save_dataframe(df, output_path)
