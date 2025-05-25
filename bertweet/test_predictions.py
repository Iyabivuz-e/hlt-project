import pandas as pd
from datasets import Dataset
from model import *
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, PeftModel
import torch # Make sure torch is imported if not already
# from sklearn.metrics import precision_recall_fscore_support # Import this if not already in the BertweetModel class
# import evaluate # Import this if not already in the BertweetModel class
import numpy as np # Import this if not already in the BertweetModel class

# Assuming BertweetModel class is defined in a previous cell and available
# (It was in the context provided)

model_path = "./results"
path_test="test_multiclass_preprocessed.csv"
model_name="vinai/bertweet-base"
test_csv=path_test
# The number of labels needs to match the number used during training.
# From main_function, this was determined by len(train_df["label"].unique())
# If you know this number is consistently 5, you can keep it.
# If it could vary, you would need to save this number during training
# and load it here. For now, assuming it's 5 based on the previous code.
num_labels=5

d_frame = pd.read_csv(test_csv).dropna(subset=["tweet_soft"])
hf_test_data = Dataset.from_pandas(d_frame) ## We wrap in the hf format for a better batch

# Initialize the model with the correct number of labels
classifier = BertweetModel(model_name=model_name, num_labels=num_labels)
print(f"My classifier here: {classifier}")

## Then we load the model.
# The load_model method should correctly initialize the base model
# with the number of labels from the class instance.
classifier.load_model(model_path)


## preprocess the data
# The map function should be applied to the dataset object itself, not to the classifier.
# The preprocess_data method of the classifier expects a dataset as input.
tokenized_test_data = classifier.preprocess_data(hf_test_data)

## Predict the output and the confidence
# Correct the variable name from tokenized_test_dat to tokenized_test_data
prediction_outputs = classifier.predict(tokenized_test_data)
logits = prediction_outputs.predictions
label_predicted = logits.argmax(axis=1)
confidence_level = logits.softmax(axis=1).max(axis=1)

d_frame["predicted_labels"] = label_predicted
d_frame["predicted_probabilities"] = confidence_level

d_frame.to_csv("test_predictions.csv", index=False)

print(f"predictions: {d_frame.heaf()}")