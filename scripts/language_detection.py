from langdetect import detect, DetectorFactory

# Set seed for reproducible language detection results
DetectorFactory.seed = 42

class LanguageDetector:
    """
    Class for detecting the language of a given text.
    """
    def __init__(self, target_lang='en'):
        """
        Initializes the LanguageDetector with a target language.

        Args:
            target_lang (str, optional): Language code to detect against.
                                        Defaults to 'en' for English.
        """
        self.target_lang = target_lang

    def is_target_language(self, text):
        """
        Checks if the input text is written in the target language.

        Args:
            text (str): Text to analyze.

        Returns:
            bool: True if text is in the target language, False otherwise.
        """
        try:
            return detect(text) == self.target_lang
        except:
            return False
