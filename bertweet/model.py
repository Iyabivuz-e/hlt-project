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

# A class that contains the finetuning of bertweet model
class BertweetModel:
    def __init__(self, num_labels, model_name):
        # Initializing the model's instance variables
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        base_model =  AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)

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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ****A function to freeze some layers and train the head******

    # def freeze_base_model(self):
    #     for params in self.model.roberta.parameters():
    #         params.requires_grad = False


    # def freeze_early_layers(self, num_layers_to_freeze=6):
    #   if hasattr(self.model, "roberta"):
    #       encoder_layers = self.model.roberta.encoder.layer
    #   elif hasattr(self.model, "bert"):  # fallback if it uses BERT-style naming
    #       encoder_layers = self.model.bert.encoder.layer
    #   else:
    #       raise ValueError("Model base not recognized. Cannot freeze layers.")

    #   for i, layer in enumerate(encoder_layers):
    #       if i < num_layers_to_freeze:
    #           for param in layer.parameters():
    #               param.requires_grad = False



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
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_attention_mask=True
        )
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
      f1_metric = evaluate.load("f1")

      # Compute accuracy
      accuracy = accuracy_metric.compute(
          predictions=predictions, references=labels)

      # Determine average type
      average_type = "weighted" if self.num_labels > 2 else "binary"

      # Compute f1, precision, recall
      precision, recall, f1, _ = precision_recall_fscore_support(
          labels, predictions, average=average_type, zero_division=0
      )

      return {
          "accuracy": accuracy["accuracy"],
          "f1_weighted": f1,
          "precision": precision,
          "recall": recall
    }


    # print("Training on device:", self.device)

    # ******* A function to finetune(train) the model*****

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

    # ******* A function to predict a single new data ******

    def predict(self, dataset):
      if self.trainer is None:
        raise ValueError("Model has not yet been trained")
      return self.trainer.predict(dataset)

    ## Predict for a single data
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

       ## ******* A function to save the model and tokenizer *****

    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    # ******* A function to load the model and tokenizer *****

    def load_model(self, model_path):
      # If your saved model has LoRA adapters
      self.model = PeftModel.from_pretrained(AutoModelForSequenceClassification.from_pretrained(model_path), model_path)
      self.tokenizer = AutoTokenizer.from_pretrained(model_path)
      self.model.to(self.device)
      self.trainer = None


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
