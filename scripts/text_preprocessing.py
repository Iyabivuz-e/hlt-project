import re
import contractions
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# !python -m spacy download en_core_web_sm (running in the notebbok for having SpaCy model for lemmatization)


# Ensure NLTK resources are downloaded
nltk.download('stopwords')

class TextPreprocessor:
    """
    Class for text preprocessing operations such as expanding contractions,
    removing emojis, normalizing characters, cleaning text, removing stopwords,
    and applying lemmatization and stemming.
    """
    def __init__(self):
        """
        Initializes the TextPreprocessor class by loading resources.
        """
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            raise LookupError("Stopwords not found.")
        
        self.stemmer = PorterStemmer()

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            raise OSError("SpaCy model 'en_core_web_sm' not found.")
        
    def _ensure_text(self, text):
        """Helper method to ensure input is string."""
        if not isinstance(text, str):
            raise TypeError(f"Expected a string, but got {type(text).__name__}")
        return text

    def expand_contractions(self, text):
        """
        Expands contractions in the input text.
        """
        text = self._ensure_text(text)
        return contractions.fix(text)

    def remove_emojis(self, text):
        """
        Removes emojis from the input text using a regular expression.
        """
        text = self._ensure_text(text)
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def normalize_repeated_chars(self, text):
        """
        Reduces repetitions to a maximum of two consecutive identical characters.
        """
        text = self._ensure_text(text)
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def remove_non_ascii(self, text):
        """
        Removes non-ASCII characters from the text.
        """
        text = self._ensure_text(text)
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def remove_stopwords(self, text):
        """
        Removes common English stop words from the text.
        """
        text = self._ensure_text(text)
        return ' '.join([word for word in text.split() if word.lower() not in self.stop_words])

    def lemmatize_text(self, text):
        """
        Applies lemmatization using SpaCy to the input text.
        """
        text = self._ensure_text(text)
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def clean_text_soft(self, text):
        """
        Soft cleaning optimized for Transformer-based models (e.g., BERT).
        Removes only superficial noise (URLs, mentions, HTML tags),
        and retains most linguistic features useful for contextual models.
        """
        text = self._ensure_text(text)
        text = re.sub(r'@\w+', '', text)                    # Remove mentions
        text = re.sub(r'http\S+|www\.\S+', '', text)        # Remove URLs
        text = re.sub(r'#', '', text)                       # Remove hashtag symbol
        text = re.sub(r'<.*?>', '', text)                   # Remove HTML tags
        text = re.sub(r'\s+', ' ', text).strip()            # Normalize whitespace
        return text
    
    def clean_text_full(self, text):
        """
        Full cleaning: starts with soft cleaning (URLs, mentions, hashtags, etc.),
        then expands contractions, normalizes, lowercases, removes stopwords, lemmatizes, and stems.
        """
        text = self._ensure_text(text)
        text = self.clean_text_soft(text)
        text = self.expand_contractions(text)
        text = self.normalize_repeated_chars(text)
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)             # Remove punctuation (keep words and whitespace)
        text = self.lemmatize_text(text)
        text = self.remove_stopwords(text)
        text = re.sub(r'\s+', ' ', text).strip()        # Normalize extra whitespace
        return text

    def preprocess(self, text, soft=True):
        """
        Preprocesses the input text using either soft or full cleaning.

        Args:
            text (str): Text to preprocess.
            soft (bool, optional): If True, applies soft cleaning only;
                                   if False, applies both soft and full cleaning. Defaults to True.

        Returns:
            str: Preprocessed text.
        """
        text = self.clean_text_soft(text)
        if not soft:
            text = self.clean_text_full(text)
        return text

