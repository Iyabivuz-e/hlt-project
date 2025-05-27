"import logistic model"
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from typing import Tuple, Union

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


############################################################
"import fasttext model"
from fast_text.load import load_fasttext

############################################################

############################################################
"import bertweet model"
from bertweet.helpers import *
from bertweet.model import *
from scripts.config import *

############################################################
"import distilroberta model"
import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, PeftModel
from transformers.modeling_outputs import SequenceClassifierOutput

############################################################
"import mamba"
from transformers import AutoTokenizer, DataCollatorWithPadding
from hf_mamba_classification import MambaForSequenceClassification
from peft import PeftModel
from types import MethodType
import torch
from load_mamba_for_predictions import get_mamba_binary
from load_mamba_for_predictions import get_mamba_multi

############################################################
"insert call to the model loader"

def load_all_models_bin():
    vectorizer = FullTextTfidfVectorizer()
    vectorizer.load(os.path.join("models/models_bin/logistic/", "vectorizer_binary.joblib"))

    model = TfidfLogisticModel()
    model.load(os.path.join("models/models_bin/logistic/", "logistic_binary.joblib"))

    logistic = model, vectorizer

    model_roberta, tok_roberta = load_model_pickle("models/models_bin/Roberta/best_model","roberta-base")
    model_distil, tok_distil = load_model_pickle("models/models_bin/DistilRoberta/best_model","distilroberta-base")
    model_mamba = get_mamba_binary()
    model_fasttext = load_fasttext("binary")

    bertweet, bertweet_tokenizer = load_bertweet_bin()


    ensemble_models = [
        (model_roberta, tok_roberta, "roberta-base"),
        (model_distil, tok_distil, "distilroberta-base"),
        (model_mamba, None, "mamba"),
        (model_fasttext, None, "fasttext"),
        (bertweet, bertweet_tokenizer, "vinai/bertweet-base"),
    ]
    return logistic, ensemble_models

def load_all_models_mul():
    vectorizer = FullTextTfidfVectorizer()
    vectorizer.load(os.path.join("models/models_mul/logistic/", "vectorizer_multiclass.joblib"))

    model = TfidfLogisticModel()
    model.load(os.path.join("models/models_mul/logistic/", "logistic_multiclass.joblib"))

    logistic = model, vectorizer

    model_roberta, tok_roberta = load_model_pickle("models/models_mul/Roberta/best_model","roberta-base")
    model_distil, tok_distil = load_model_pickle("models/models_mul/DistilRoberta/best_model","distilroberta-base")
    model_mamba = get_mamba_multi()
    model_fasttext = load_fasttext("multiclass")
    bertweet, bertweet_tokenizer = load_bertweet_multi()

    ensemble_models = [
        (model_roberta, tok_roberta, "roberta-base"),
        (model_distil, tok_distil, "distilroberta-base"),
        (model_mamba, None, "mamba"),
        (model_fasttext, None, "fasttext"),
        (bertweet, bertweet_tokenizer, "vinai/bertweet-base"),

    ]
    return logistic, ensemble_models

############################################################
"logistic model loader"

class FullTextTfidfVectorizer:
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit_transform(self, df: pd.DataFrame):
        texts = df["tweet_full"].fillna("")
        return self.vectorizer.fit_transform(texts)

    def transform(self, df: pd.DataFrame):
        texts = df["tweet_full"].fillna("")
        return self.vectorizer.transform(texts)

    def transform_phrase(self, text):
        return self.vectorizer.transform([text])

    def save(self, path: str):
        joblib.dump(self.vectorizer, path)

    def load(self, path: str):
        self.vectorizer = joblib.load(path)


class TfidfLogisticModel:
    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        class_weight: Union[str, dict, None] = None,
        solver: str = "liblinear",
        multi_class: str = "ovr"
    ):
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            solver=solver,
            multi_class=multi_class,
            max_iter=1000
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y_true):
        y_pred = self.model.predict(X)
        return classification_report(y_true, y_pred)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

############################################################
"distilroberta model loader"

class CustomClassifier(nn.Module):
    def __init__(self, model_name, config, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        base = AutoModel.from_pretrained(model_name, config=config)
        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none", target_modules=["query", "value"]
        )
        self.base_model = get_peft_model(base, lora_cfg)

        # custom head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size//2),
            nn.LayerNorm(config.hidden_size//2),
            nn.GELU(),
            nn.Linear(config.hidden_size//2, config.num_labels)
        )
        # init head
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None))

def load_model_pickle(path, model_name, device="cpu"):
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = CustomClassifier(model_name, config)
    model.base_model.load_adapter(path, adapter_name="default")
    classifier_path = os.path.join(path, "classifier.pt")
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

    return model, tokenizer

############################################################
"Insert model loader for each model"
############################################################
"Insert model loader for each model"
