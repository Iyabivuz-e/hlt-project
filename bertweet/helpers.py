import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model import *


## For multiclass classification
def load_bertweet_multi(model_path="./results/merged_results_multiclass", num_labels=5):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    
    classifier = BertweetModelMulticlass(num_labels=num_labels, model_name="vinai/bertweet-base")
    classifier.model = model
    classifier.tokenizer = tokenizer
    classifier.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.model.to(classifier.device)

    return classifier, tokenizer


### For binary classification
def load_bertweet_bin(model_path="./results/merged_results_binary", num_labels=2):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    classifier = BertweetModelBinary(num_labels=num_labels, model_name="vinai/bertweet-base")
    classifier.model = model
    classifier.tokenizer = tokenizer
    classifier.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.model.to(classifier.device)

    return classifier, tokenizer
