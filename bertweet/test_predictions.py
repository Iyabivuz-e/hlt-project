import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer # Import Trainer
import torch
import numpy as np
from peft import PeftModel
from transformers import TrainingArguments, DataCollatorWithPadding 
from model import *

# Assuming BertweetModel class is defined in a previous cell and available

model_path = "./results"
path_test="test_multiclass_preprocessed.csv"
model_name="vinai/bertweet-base"
test_csv=path_test
num_labels=5

d_frame = pd.read_csv(test_csv).dropna(subset=["tweet_soft"])
hf_test_data = Dataset.from_pandas(d_frame)

# Initialize the model with the correct number of labels
classifier = BertweetModel(model_name=model_name, num_labels=num_labels)
print(f"My classifier here: {classifier}")

# Load the trained PEFT model and tokenizer
classifier.load_model(model_path)

## Preprocess the data
tokenized_test_data = classifier.preprocess_data(hf_test_data)

# Re-initialize the Trainer for prediction/evaluation after loading the model
# We need TrainingArguments and DataCollatorWithPadding for this.
# These don't need to be the exact same as training, but they should be compatible.
# We can use a minimal TrainingArguments config for inference.
inference_args = TrainingArguments(
    output_dir="./inference_results", 
    per_device_eval_batch_size=16,
    dataloader_drop_last=False,
    report_to="none", # No reporting needed for inference
)

data_collator = DataCollatorWithPadding(tokenizer=classifier.tokenizer)

# Create a new Trainer instance
classifier.trainer = Trainer(
    model=classifier.model,
    args=inference_args,
    compute_metrics=classifier.compute_metrics, # Keep the same compute_metrics function
    tokenizer=classifier.tokenizer,
    data_collator=data_collator,
)


## Predict the output and the confidence
prediction_outputs = classifier.predict(tokenized_test_data)
logits = prediction_outputs.predictions
label_predicted = logits.argmax(axis=1)

# Make sure logits are a torch tensor before softmax
confidence_scores = torch.softmax(torch.tensor(logits), dim=1)
# Get the probability of the predicted class for each sample
confidence_level, _ = confidence_scores.max(dim=1)


d_frame["predicted_labels"] = label_predicted
d_frame["predicted_probabilities"] = confidence_level.numpy()

d_frame.to_csv("test_predictions.csv", index=False)