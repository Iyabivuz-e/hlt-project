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

    # merge_and_save_model method 
    def merge_and_save_model(self, save_path):
        """
        Merge LoRA weights with base model and save
        """
        try:
            print(f"Starting merge process...")
            print(f"Save path: {save_path}")
            
            # Check if model is actually a PEFT model
            from peft import PeftModel
            if not isinstance(self.model, PeftModel):
                print("Warning: Model is not a PEFT model, cannot merge")
                # Just save the regular model instead
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"Regular model saved to {save_path}")
                return
            
            # Creating a directory if it doesn't exist
            import os
            os.makedirs(save_path, exist_ok=True)
            print(f"Created directory: {save_path}")
            
            # Merge and unload
            print("Merging LoRA weights with base model...")
            merged_model = self.model.merge_and_unload()
            print("Merge completed successfully")
            
            # Save merged model
            print("Saving merged model...")
            merged_model.save_pretrained(save_path)
            print("Merged model saved")
            
            # Save tokenizer
            print("Saving tokenizer...")
            self.tokenizer.save_pretrained(save_path)
            print("Tokenizer saved")
            
            print(f"Successfully merged and saved model to: {save_path}")
            
            # Verify files were created
            files = os.listdir(save_path)
            print(f"Files in save directory: {files}")
            
        except Exception as e:
            print(f"Error during merge and save: {e}")
            import traceback
            traceback.print_exc()
            

    def load_model(self, model_path):
        base_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self.num_labels)
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        self.trainer = None

    def plot_confusion_matrix(self, dataset):
        """Plot confusion matrix for evaluation dataset"""
        try:
            if self.trainer is None:
                print("Warning: No trainer available for confusion matrix")
                return
            
            predictions = self.trainer.predict(dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            
            plt.figure(figsize=(8, 6))
            disp.plot(cmap='Blues')
            plt.title('Confusion Matrix')
            plt.show()
            
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")

    def print_metrics_summary(self, eval_results):
        """summary of evaluation metrics"""
        try:
            print("\n" + "="*50)
            print("EVALUATION RESULTS SUMMARY")
            print("="*50)
            
            for key, value in eval_results.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Error printing metrics summary: {e}")


class BertweetModelBinary(BaseBertweetModel):
    def __init__(self, num_labels=2, model_name="vinai/bertweet-base"):
        super().__init__(num_labels=num_labels, model_name=model_name)


class BertweetModelMulticlass(BaseBertweetModel):
    def __init__(self, num_labels=5, model_name="vinai/bertweet-base"):
        super().__init__(num_labels=num_labels, model_name=model_name)

