from logging import raiseExceptions
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np
from datasets import Dataset
import evaluate
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# In bertweet.py or bertweet_models.py

class BaseBertweetModel:
    def __init__(self, num_labels, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        self.model = get_peft_model(base_model, lora_config)
        self.trainer = None
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize_function(self, examples):
        texts = [str(x) for x in examples["tweet_soft"]]
        encoder = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_attention_mask=True
        )
        return encoder

    def preprocess_data(self, raw_dataset):
        return raw_dataset.map(self.tokenize_function, batched=True)

    def compute_metrics(self, evaluate_predictions):
        logits, labels = evaluate_predictions
        predictions = np.argmax(logits, axis=1)

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        average_type = "weighted" if self.num_labels > 2 else "binary"

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average_type, zero_division=0
        )
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

        return {
            "accuracy": accuracy["accuracy"],
            "f1_weighted": f1,
            "precision": precision,
            "recall": recall
        }

    def train(self, train_dataset, eval_dataset, training_args, callbacks, data_collator):
        self.trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
            data_collator=data_collator
        )
        self.trainer.train()

    def evaluate(self, dataset):
        if self.trainer is None:
            raise ValueError("Model not trained yet")
        return self.trainer.evaluate(eval_dataset=dataset)

    def predict_single(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return torch.argmax(probs), probs

    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load_model(self, model_path):
        base_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self.num_labels)
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.trainer = None


class BertweetModelBinary(BaseBertweetModel):
    def __init__(self, num_labels=2, model_name="vinai/bertweet-base"):
        super().__init__(num_labels=num_labels, model_name=model_name)


class BertweetModelMulticlass(BaseBertweetModel):
    def __init__(self, num_labels=5, model_name="vinai/bertweet-base"):
        super().__init__(num_labels=num_labels, model_name=model_name)

