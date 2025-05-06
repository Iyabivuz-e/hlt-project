import pandas as pd
import re

class DataUnderstanding:
    """
    Class for generating an overview and basic statistics of a cyberbullying dataset.
    """

    def __init__(self, dataset: pd.DataFrame, text_column: str = 'tweet_text', class_column: str = 'cyberbullying_type'):
        if dataset.empty:
            raise ValueError("The provided dataset is empty.")
        self.dataset = dataset
        self.text_column = text_column
        self.class_column = class_column

    def class_distribution(self):
        """Prints the number of examples per class."""
        try:
            print("\nClass Distribution:")
            print(self.dataset[self.class_column].value_counts())
        except KeyError:
            print(f"Column '{self.class_column}' not found in the dataset.")

    def check_imbalance(self):
        """Prints imbalance ratio between most and least frequent class."""
        try:
            counts = self.dataset[self.class_column].value_counts()
            imbalance_ratio = counts.max() / counts.min()
            print(f"\nClass Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
        except KeyError:
            print(f"Column '{self.class_column}' not found in the dataset.")
        except ZeroDivisionError:
            print("Cannot compute imbalance ratio: division by zero.")

    def check_missing_values(self):
        """Prints the number of missing values for each column."""
        print("\nMissing Values:")
        print(self.dataset.isnull().sum())

    def check_empty_strings(self):
        """Checks and prints the number of empty or whitespace-only strings per column."""
        try:
            empty_counts = (self.dataset
                            .applymap(lambda x: isinstance(x, str) and x.strip() == '')
                            .sum())
            print("\nEmpty or whitespace-only strings per column:")
            print(empty_counts)
        except Exception as e:
            print(f"Error while checking empty strings: {e}")

    def check_duplicates(self):
        """Prints the number of duplicated entries based on the text column."""
        try:
            num_duplicates = self.dataset[self.text_column].duplicated().sum()
            print(f"\nNumber of duplicated {self.text_column}: {num_duplicates}")
        except KeyError:
            print(f"Column '{self.text_column}' not found in the dataset.")

    def inspect_duplicates(self, text_column: str = None, label_column: str = None):
        """
        Analyzes duplicated texts, separates perfect and imperfect duplicates.
        """
        text_column = text_column or self.text_column
        label_column = label_column or self.class_column
        try:
            duplicates_all = self.dataset[self.dataset.duplicated(subset=[text_column], keep=False)]
            print(f"\nTotal duplicated texts (same text, any label): {duplicates_all.shape[0]} rows")

            label_counts_in_duplicates = duplicates_all[label_column].value_counts()
            print("\nLabel counts among all duplicates:")
            for label, count in label_counts_in_duplicates.items():
                print(f"{label}: {count}")

            duplicates_perfect = self.dataset[self.dataset.duplicated(subset=[text_column, label_column], keep=False)]
            text_keys_perfect = set(duplicates_perfect[text_column])
            text_keys_all = set(duplicates_all[text_column])
            text_keys_imperfect = text_keys_all - text_keys_perfect

            duplicates_imperfect = duplicates_all[duplicates_all[text_column].isin(text_keys_imperfect)]
            print(f"\nPerfect duplicates (same text and same label): {duplicates_perfect.shape[0]} rows")
            print(f"Imperfect duplicates (same text, different labels): {duplicates_imperfect.shape[0]} rows")
        except KeyError as e:
            print(f"Column not found: {e}")

    def average_tweet_length(self):
        """Computes and prints the average tweet length in characters and words."""
        try:
            self.dataset['char_length'] = self.dataset[self.text_column].astype(str).apply(len)
            self.dataset['word_length'] = self.dataset[self.text_column].astype(str).apply(lambda x: len(x.split()))
            print(f"\nAverage Tweet Length: {self.dataset['char_length'].mean():.2f} characters")
            print(f"Average Tweet Length: {self.dataset['word_length'].mean():.2f} words")
        except KeyError:
            print(f"Column '{self.text_column}' not found in the dataset.")

    def hashtag_analysis(self):
        """Counts and prints the average number of hashtags per tweet."""
        try:
            hashtag_counts = self.dataset[self.text_column].astype(str).apply(lambda x: len(re.findall(r"#\w+", x)))
            avg_hashtags = hashtag_counts.mean()
            print(f"\nAverage Hashtags per Tweet: {avg_hashtags:.2f}")
        except KeyError:
            print(f"Column '{self.text_column}' not found in the dataset.")

    def emoji_analysis(self):
        """Counts and prints the average number of emojis per tweet."""
        emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
        try:
            emoji_counts = self.dataset[self.text_column].astype(str).apply(lambda x: len(emoji_pattern.findall(x)))
            avg_emojis = emoji_counts.mean()
            print(f"\nAverage Emojis per Tweet: {avg_emojis:.2f}")
        except KeyError:
            print(f"Column '{self.text_column}' not found in the dataset.")

    def binary_class_distribution(self, binary_column: str = 'is_cyberbullying'):
        """Prints the distribution of binary classes if available."""
        if binary_column in self.dataset.columns:
            print("\nBinary Class Distribution:")
            print(self.dataset[binary_column].value_counts(normalize=True).rename({0: 'No', 1: 'Yes'}))
        else:
            print(f"Column '{binary_column}' not found in the dataset.")

    def full_overview(self):
        """Runs all analysis methods to provide a complete dataset overview."""
        self.class_distribution()
        self.check_imbalance()
        self.check_missing_values()
        self.check_empty_strings()
        self.check_duplicates()
        self.average_tweet_length()
        self.hashtag_analysis()
        self.emoji_analysis()
        self.binary_class_distribution()
        self.inspect_duplicates()
