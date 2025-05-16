import pandas as pd

class DataCleaner:
    """
    A class for cleaning a dataset by removing duplicates and missing values,
    handling one or more text columns and a label column.
    """

    def __init__(self, df: pd.DataFrame, text_column, label_column: str = 'cyberbullying_type'):
        if df.empty:
            raise ValueError("The provided dataset is empty.")

        if isinstance(text_column, str):
            self.text_columns = [text_column]
        elif isinstance(text_column, list):
            self.text_columns = text_column
        else:
            raise TypeError("text_column must be a string or list of strings.")

        self.df = df
        self.label_column = label_column

    def clean_text_duplicates(self) -> pd.DataFrame:
        """
        Removes text duplicates across multiple text columns:
        - Removes rows with same text but different labels (imperfect duplicates)
        - Keeps only one row for same text and same label (perfect duplicates)
        """
        print(f"\n--- CLEANING DUPLICATES COLUMN BY COLUMN: {self.text_columns} ---")
        before_total = self.df.shape[0]
        rows_to_keep = pd.Series([True] * len(self.df), index=self.df.index)

        for text_col in self.text_columns:
            print(f"\nProcessing column: '{text_col}'")

            # 1. Trova tutti i duplicati (testo ripetuto in quella colonna)
            duplicates_all = self.df[self.df.duplicated(subset=[text_col], keep=False)]

            # 2. Identifica duplicati imperfetti (stesso testo, label diversa)
            duplicates_imperfect = duplicates_all[
                self.df.duplicated(subset=[text_col], keep=False) &
                self.df.duplicated(subset=[text_col, self.label_column], keep=False) == False
            ]
            conflicted_texts = duplicates_imperfect[text_col].unique().tolist()

            # 3. Rimuovi tutte le righe con quei testi ambigui
            mask_conflict = self.df[text_col].isin(conflicted_texts)
            rows_to_keep &= ~mask_conflict

            # 4. Tra i duplicati perfetti, tieni solo la prima occorrenza
            df_temp = self.df[~mask_conflict].copy()
            duplicated_perfect_mask = df_temp.duplicated(subset=[text_col, self.label_column], keep='first')
            rows_to_keep[df_temp[duplicated_perfect_mask].index] = False

            print(f" - Removed {mask_conflict.sum()} imperfect duplicates (conflicting labels)")
            print(f" - Removed {duplicated_perfect_mask.sum()} perfect duplicates (keeping one)")

        # Applica la maschera finale
        self.df = self.df[rows_to_keep].reset_index(drop=True)
        after_total = self.df.shape[0]
        print(f"\nTotal rows removed: {before_total - after_total}")
        print("--- DUPLICATE CLEANING COMPLETED ---")
        return self.df


    def drop_missing_values(self, important_columns: list) -> pd.DataFrame:
        """
        Drop rows from the DataFrame that contain missing values in the specified columns.
        This method checks whether the specified columns exist, removes any rows with
        missing (NaN) values in those columns, and prints the number of rows removed.
        """
        print(f"\n--- DROPPING MISSING VALUES IN: {important_columns} ---")
        missing_cols = [col for col in important_columns if col not in self.df.columns]
        if missing_cols:
            raise KeyError(f"The following columns are missing in the DataFrame: {missing_cols}")

        before = self.df.shape[0]
        self.df = self.df.dropna(subset=important_columns).reset_index(drop=True)
        after = self.df.shape[0]
        print(f"Removed {before - after} rows with missing values.")
        print("\nMISSING VALUE CLEANING COMPLETED.")

        return self.df

