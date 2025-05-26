# Cyberbullying Detection Project

## Overview

This project implements a pipeline for detecting cyberbullying in social media text. It covers data ingestion, cleaning, preprocessing, exploratory analysis, dataset splitting, and saving ready-to-use train/validation/test splits.

---

## Project Structure

```
cyberbullying/
├── data/
│   ├── raw_data/            # Original dataset (unmodified)
│   ├── interim/             # Train/val/test splits
│   └── processed_data/      # Cleaned, labeled, preprocessed data + label mapping
├── notebooks/
│   ├── data_cleaning.ipynb       # Exploratory cleaning
│   ├── data_understanding.ipynb  # Dataset statistics
│   └── workflow.ipynb            # Main pipeline execution
├── scripts/
│   ├── config.py              # Project-wide constants
│   ├── data_loader.py         # Dataset loading
│   ├── data_understanding.py  # Exploratory analysis tools
│   ├── text_preprocessing.py  # Full/soft text cleaning functions
│   ├── language_detection.py  # English filtering using langdetect
│   ├── data_splitting.py      # Train/val/test splitting with stratification
│   ├── data_saver.py          # Save DataFrame or splits
│   ├── data_builder.py        # Adds labels and saves label map
│   └── outputs/               # Model outputs and temporary results
└── requirements.txt           # Python dependencies
```

---

## Workflow (notebook: `workflow.ipynb`)

1. **Project Setup**

   * Set base path
   * Import modules and config

2. **Load Dataset**

   * Load raw CSV using `DataLoader`

3. **Data Understanding**

   * Show class distribution
   * Check for imbalance
   * Explore tweet length, hashtags, emoji frequency

4. **Language Filtering**

   * Filter only English tweets using `langdetect`

5. **Handling Class Imbalance**

   * Remove duplicate samples
   * Apply text data augmentation (e.g., EDA: synonyms replacement, random insertions)
   * Experiment with class weighting during training

6. **Feature Extraction**  (for baseline and NN)

   * Bag of Words (BoW)
   * TF-IDF

7. **Baseline Models**

   * Logistic Regression
   * Naive Bayes (Bernoulli or Multinomial)

8. **Advanced Models**

   * Fine-tune Transformer models:

     * `bert-base-uncased`
     * `cardiffnlp/twitter-roberta-base-offensive`
     * BERT-ADA
     * BERTweet-base
     * RoBERTa fine-tuned on hate speech subtypes
     * RoBERTa
   * Other architectures: FastText, Mamba

9. **Ensemble Models**

   * Combine model predictions via arithmetic mean (tie → favor non-cyberbullying)
   * Ridge Regression meta-learner
   * Pipeline approach: sequentially query models based on confidence threshold or until final model produces output

10. **Model Evaluation**

* Binary classification: recall for the "non-hate" class (minimize false positives)
* Multiclass classification: accuracy, macro & micro precision, macro F1-score

---

## Preprocessing Logic

### `tweet_soft`

* Keeps linguistic richness for Transformers
* Removes: `@mentions`, URLs, `#` symbols, HTML tags, excessive whitespace

### `tweet_full`

* For TF-IDF, BoW, RNNs
* Expands contractions
* Normalizes repeated characters
* Lowercases text
* Removes stopwords
* Lemmatizes & stems
* Removes punctuation and cleans spacing

---

## Labeling

* **Binary**: `0` if “not\_cyberbullying”, else `1`
* **Multiclass**: encoded using `LabelEncoder`; mapping saved as `label_mapping.json`

---

## Outputs

* `dataset_preprocessed.csv`: full preprocessed dataset with all versions and labels
* `train.csv`, `val.csv`, `test.csv`: stratified splits ready for model training
* `label_mapping.json`: class-to-index mapping for model output interpretation

---

## Notes

* Use `tweet_soft` for Transformer-based models
* Use `tweet_full` for traditional models (Logistic Regression, Naive Bayes, LSTM)
* Preprocessing is modularized in `TextPreprocessor`
* Run everything step-by-step from `workflow.ipynb`

---

## Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Required packages include:

* `pandas`
* `scikit-learn`
* `spacy`
* `nltk`
* `contractions`
* `langdetect`
* `transformers` (if using BERT-based models)

