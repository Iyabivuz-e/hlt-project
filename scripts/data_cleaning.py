import pandas as pd

class DataCleaner:
    """
    A class for fully cleaning a dataset by removing all types of duplicate entries.

    Steps performed:
    - Remove perfect duplicates (same text and label)
    - Remove conflicting duplicates (same text, different labels)
    - Remove pure text duplicates
    """

    def __init__(self, df: pd.DataFrame, text_column: str = 'tweet_text', label_column: str = 'cyberbullying_type'):
        """
        Initializes the DataCleaner with the given DataFrame and column names.

        Args:
            df (pd.DataFrame): Input dataset containing text and labels.
            text_column (str): Name of the column containing text.
            label_column (str): Name of the column containing labels.
        """
        if df.empty:
            raise ValueError("The provided dataset is empty.")
        self.df = df
        self.text_column = text_column
        self.label_column = label_column

    def clean_all_duplicates(self) -> pd.DataFrame:
        """
        Perform full cleaning by sequentially removing:
        1. Perfect duplicates (identical text and label)
        2. Conflicting duplicates (same text assigned to different labels)
        3. Pure text duplicates (identical texts regardless of label)

        Returns:
            pd.DataFrame: Cleaned dataset with duplicates removed.
        """
        print(f"\n--- CLEANING DUPLICATES BASED ON TEXT COLUMN: {self.text_column} ---")

        try:
            # Step 1: Remove perfect duplicates
            print("\n[1/3] Removing perfect duplicates...")
            before = self.df.shape[0]
            self.df = self.df.drop_duplicates(subset=[self.text_column, self.label_column]).reset_index(drop=True)
            after = self.df.shape[0]
            print(f"Removed {before - after} perfect duplicates.")

            # Step 2: Remove conflicting label duplicates
            print("\n[2/3] Removing conflicting label duplicates...")
            before = self.df.shape[0]
            conflict_texts = (
                self.df.groupby(self.text_column)[self.label_column]
                .nunique()
                .reset_index()
                .query(f"{self.label_column} > 1")[self.text_column]
                .tolist()
            )
            self.df = self.df[~self.df[self.text_column].isin(conflict_texts)].reset_index(drop=True)
            after = self.df.shape[0]
            print(f"Removed {before - after} conflicting label rows.")

            # Step 3: Remove pure text duplicates
            print("\n[3/3] Forcing final text-only duplicate removal...")
            before = self.df.shape[0]
            self.df = self.df.drop_duplicates(subset=[self.text_column]).reset_index(drop=True)
            after = self.df.shape[0]
            print(f"Removed {before - after} pure text duplicates.")

            print("\nDUPLICATE CLEANING COMPLETED.")

        except KeyError as e:
            print(f"Column not found during duplicate cleaning: {e}")

        return self.df
    
    def drop_missing_values(self, important_columns: list) -> pd.DataFrame:
        """
        Remove rows with missing values in specified important columns, 
        to ensure quality inputs for model training.
        
        Args:
            important_columns (list): List of column names that must not contain missing values.

        Returns:
            pd.DataFrame: Cleaned dataset with missing values removed in important columns.
        """
        print(f"\n--- DROPPING MISSING VALUES IN IMPORTANT COLUMNS: {important_columns} ---")
        missing_cols = [col for col in important_columns if col not in self.df.columns]
        if missing_cols:
            raise KeyError(f"The following important columns are missing from the DataFrame: {missing_cols}")

        before = self.df.shape[0]
        self.df = self.df.dropna(subset=important_columns).reset_index(drop=True)
        after = self.df.shape[0]
        print(f"Removed {before - after} rows with missing values.")
        print("\nMISSING VALUES CLEANING COMPLETED.")
        return self.df