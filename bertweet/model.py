from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, glue_compute_metrics
import numpy as np
import torch
from helpers import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


model_name = "vinai/bertweet-base"

class BertweetModel:
    def __init__(self, model_name=model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess(self, example):
        return self.tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    
    def comput_metrics(self, predict):
        preds = np.argmax(predict.predictions, axis=1)
        labels = predict.label_ids # will hsvr to  of the labels
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }
       