from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class SemanticSimilarity:
    """
    A class to compute semantic similarity between text entries
    using either Sentence-BERT (default) or Word2Vec embeddings.
    """

    def __init__(self, model_type='sentence-bert', model_path='all-MiniLM-L6-v2'):
        """
        Initialize the similarity model.

        Parameters:
        - model_type (str): Either 'sentence-bert' or 'word2vec'.
        - model_path (str): Path or name of the pre-trained model.
        """
        self.model_type = model_type

        if model_type == 'sentence-bert':
            self.model = SentenceTransformer(model_path)  # Load a Sentence-BERT model
        elif model_type == 'word2vec':
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)  # Load Word2Vec model
        else:
            raise ValueError("model_type must be 'sentence-bert' or 'word2vec'")

    def compute_embeddings(self, texts):
        """
        Compute embeddings for a list of texts.

        Parameters:
        - texts (List[str]): Input text data.

        Returns:
        - np.ndarray: Embeddings matrix.
        """
        if self.model_type == 'sentence-bert':
            return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)

        elif self.model_type == 'word2vec':
            embeddings = []
            for text in texts:
                tokens = text.split()
                # Compute mean vector for all words in the text that are in the vocabulary
                vectors = [self.model[word] for word in tokens if word in self.model]
                if vectors:
                    embeddings.append(np.mean(vectors, axis=0))
                else:
                    # If none of the words are in the vocabulary, return a zero vector
                    embeddings.append(np.zeros(self.model.vector_size))
            return np.array(embeddings)

    def find_most_similar(self, embeddings_source, embeddings_target):
        """
        For each source embedding, find the most similar target embedding.

        Parameters:
        - embeddings_source (np.ndarray): Embeddings of source texts.
        - embeddings_target (np.ndarray): Embeddings of target texts.

        Returns:
        - most_similar_indices (np.ndarray): Indices of the most similar targets.
        - most_similar_scores (np.ndarray): Similarity scores of the most similar pairs.
        """
        similarity_matrix = cosine_similarity(embeddings_source, embeddings_target)
        most_similar_indices = np.argmax(similarity_matrix, axis=1)
        most_similar_scores = np.max(similarity_matrix, axis=1)
        return most_similar_indices, most_similar_scores

    def predict_class(self, df_source, df_target,
                      source_text_col='tweet_text',
                      target_text_col='tweet_text',
                      target_class_col='target',
                      threshold=0.8):
        """
        Predict the most semantically similar class from the target dataframe
        for each entry in the source dataframe, based on a similarity threshold.

        Parameters:
        - df_source (pd.DataFrame): Source dataframe containing texts to classify.
        - df_target (pd.DataFrame): Target dataframe with reference texts and their classes.
        - source_text_col (str): Column name for text in df_source.
        - target_text_col (str): Column name for text in df_target.
        - target_class_col (str): Column name for class label in df_target.
        - threshold (float): Minimum similarity score to accept a match.

        Returns:
        - pd.DataFrame: Resulting dataframe with predicted classes and similarity info.
        """
        texts_source = df_source[source_text_col].tolist()
        texts_target = df_target[target_text_col].tolist()

        embeddings_source = self.compute_embeddings(texts_source)
        embeddings_target = self.compute_embeddings(texts_target)

        indices, scores = self.find_most_similar(embeddings_source, embeddings_target)
        predicted_classes = df_target.iloc[indices][target_class_col].values

        # Build the results dataframe
        results = pd.DataFrame({
            'tweet_text': df_source[source_text_col].values,
            'predicted_class': predicted_classes,
            'similarity_score': scores,
            'most_similar_target_text': df_target.iloc[indices][target_text_col].values
        })

        # Filter by threshold
        results = results[results['similarity_score'] >= threshold]
        return results
