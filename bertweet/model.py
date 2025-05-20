from logging import raiseExceptions
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np
from datasets import Dataset
import evaluate
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# A class that contains the finetuning of bertweet model
class BertweetModel:
    def __init__(self, num_labels, model_name):
        # Initializing the model's instance variables
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        self.trainer = None
        self.num_labels = num_labels
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ****A function to freeze some layers and train the head******

    def freeze_base_model(self):
        for params in self.model.roberta.parameters():
            params.requires_grad = False

    # *******A function to load the dataset and then convert it into the huggingface standard*******

    def load_dataset(self, df_path, df_format="csv"):
        if df_format == "csv":
            dataset = Dataset.from_pandas(pd.read_csv(df_path))
        else:
            raise ValueError(f"unsupported data format: {df_format}")
        return dataset

    # *******A function to tokenize the dataset******

    def tokenize_function(self, examples):
        texts = [str(x) for x in examples["tweet_soft"]]
        encoder = self.tokenizer(
            texts, truncation=True, padding="max_length", max_length=128)
        encoder["labels"] = [int(x) for x in examples["label"]]

        return encoder

    # ***** A function to tokenize the dataset and then preprocess it*****

    def preprocess_data(self, raw_dataset):
        tokenized_datasets = raw_dataset.map(
            self.tokenize_function, batched=True)
        return tokenized_datasets

    # *******function to compute the metrics for the model********

    def compute_metrics(self, evaluate_predictions):
        logits, labels = evaluate_predictions
        predictions = np.argmax(logits, axis=1)

        accuracy_metric = evaluate.load("accuracy")
        accuracy = accuracy_metric.compute(
            predictions=predictions, references=labels)

        # Check if the task is binary or multinomial then load more metrics methods
        if self.num_labels > 2:
            f1_metric = evaluate.load("f1")
            f1 = f1_metric.compute(
                predictions=predictions, references=labels, average="weighted")
            return {
                "accuracy": accuracy["accuracy"], "f1_weighted": f1["f1"]
            }
        else:
            f1_metric = evaluate.load("f1")
            f1 = f1_metric.compute(
                predictions=predictions, references=labels)
            return {
                "accuracy": accuracy["accuracy"], "f1_weighted": f1["f1"]
            }


    # ******* A function to finetune(train) the model*****

    def train(self, train_dataset, eval_dataset, training_args):
        self.trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )
        self.trainer.train()

    # ******* A function to predict a single new data ******

    def predict(self, dataset):
      if self.trainer is None:
        raise ValueError("Model has not yet been trained")
      return self.trainer.predict(dataset)

    # Predict for a single data
    def predict_single(self, text):
      inputs = self.tokenizer(
          text, truncation=True, padding=True, return_tensors="pt"
      ).to(self.device)

      with torch.no_grad():
        outputs = self.model(**inputs)
      probs = torch.softmax(outputs.logits, dim=1)
      return torch.argmax(probs), probs

    # ***** A function to evaluate the model's performance *****

    def evaluate(self, dataset):
        if self.trainer is None:
            raise ValueError("Model has not been trained yet...")
        return self.trainer.evaluate(eval_dataset=dataset)

   # ****** A function to print the metrics summary

    def print_metrics_summary(self, eval_results):
      print("\n Evaluation Metrics:")
      for k, v in eval_results.items():
          print(f"{k}: {v:.4f}")

     # **** A function to display the confusion matrix ****

    def plot_confusion_matrix(self, dataset):
      predictions = self.trainer.predict(dataset)
      preds = np.argmax(predictions.predictions, axis=1)
      labels = predictions.label_ids

      cm = confusion_matrix(labels, preds)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(cmap="Blues")
      plt.show()
