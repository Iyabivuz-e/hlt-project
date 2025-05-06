import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

import random
import pandas as pd
from nltk.corpus import wordnet, stopwords

class DataBalancer:
    """
    A class to perform data augmentation for balancing imbalanced text datasets.

    Techniques used:
    - Synonym replacement
    - Random insertion of synonyms
    - Random swapping of words
    - Random deletion of words

    Attributes:
        alpha_sr (float): Proportion of words to replace with synonyms.
        alpha_ri (float): Proportion of words where synonyms will be inserted.
        alpha_rs (float): Number of random word swaps.
        alpha_rd (float): Probability of deleting each word.
        num_aug (int): Number of augmented sentences per original sentence.
        target_size (int): Desired size of the minority class after augmentation.
        stop_words (set): Set of English stop words.
    """

    def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1, num_aug=4, target_size=15000):
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.alpha_rd = alpha_rd
        self.num_aug = num_aug
        self.target_size = target_size
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            raise LookupError("NLTK stopwords resource not found.")

    def get_synonyms(self, word):
        """Retrieve synonyms for a given word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                if synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)

    def synonym_replacement(self, words, n):
        """Replace 'n' non-stop words in the sentence with their synonyms."""
        new_words = words.copy()
        candidates = [word for word in words if word not in self.stop_words]
        random.shuffle(candidates)
        num_replaced = 0
        for word in candidates:
            synonyms = self.get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        return new_words

    def random_insertion(self, words, n):
        """Randomly insert 'n' synonyms into the sentence."""
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, words):
        """Helper function to insert a random synonym into a random position."""
        synonyms = []
        counter = 0
        while not synonyms and counter < 10:
            word = random.choice(words)
            synonyms = self.get_synonyms(word)
            counter += 1
        if synonyms:
            synonym = random.choice(synonyms)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, synonym)

    def random_swap(self, words, n):
        """Randomly swap two words in the sentence 'n' times."""
        new_words = words.copy()
        if len(new_words) < 2:
            return new_words  # Cannot swap if less than two words
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words

    def random_deletion(self, words, p):
        """Randomly delete each word in the sentence with probability 'p'."""
        if len(words) == 1:
            return words  # Cannot delete the only word
        return [word for word in words if random.uniform(0,1) > p]

    def augment_sentence(self, sentence):
        """
        Apply all augmentation methods to a single sentence.

        Returns:
            List of augmented sentences.
        """
        words = sentence.split()
        if len(words) < 4:  # Sentences that are too short are not augmented
            return [sentence]
        num_words = len(words)
        augmented_sentences = []

        n_sr = max(1, int(self.alpha_sr * num_words))
        n_ri = max(1, int(self.alpha_ri * num_words))
        n_rs = max(1, int(self.alpha_rs * num_words))

        augmented_sentences.append(' '.join(self.synonym_replacement(words, n_sr)))
        augmented_sentences.append(' '.join(self.random_insertion(words, n_ri)))
        augmented_sentences.append(' '.join(self.random_swap(words, n_rs)))
        augmented_sentences.append(' '.join(self.random_deletion(words, self.alpha_rd)))

        return augmented_sentences

    def balance(self, df, text_column='text', label_column='label', minority_label=0):
        """
        Balance the dataset by augmenting the minority class to the target size.

        Args:
            df (DataFrame): Input dataset with text and labels.
            text_column (str): Name of the text column.
            label_column (str): Name of the label column.
            minority_label (any): Label value corresponding to the minority class.

        Returns:
            DataFrame: A new balanced dataset.
        """
        if text_column not in df.columns or label_column not in df.columns:
            raise KeyError(f"Columns '{text_column}' or '{label_column}' not found in DataFrame.")
        
        df_minority = df[df[label_column] == minority_label]
        df_majority = df[df[label_column] != minority_label]
        
        if df_minority.empty:
            raise ValueError(f"No samples found with label {minority_label}.")

        augmented_texts = []
        augmented_labels = []

        for _, row in df_minority.iterrows():
            sentence = row[text_column]
            augmented_sentences = self.augment_sentence(sentence)
            for aug_sentence in augmented_sentences:
                augmented_texts.append(aug_sentence)
                augmented_labels.append(minority_label)
                if len(augmented_texts) + len(df_minority) >= self.target_size:
                    break
            if len(augmented_texts) + len(df_minority) >= self.target_size:
                break

        df_augmented = pd.DataFrame({text_column: augmented_texts, label_column: augmented_labels})
        df_balanced = pd.concat([df_majority, df_minority, df_augmented]).sample(frac=1).reset_index(drop=True)
        return df_balanced