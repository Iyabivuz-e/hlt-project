from model import *

def load_bertweet_binary():
    bertweet = BertweetModelBinary(num_labels=2, model_name="vinai/bertweet-base")
    bertweet.load_model("models/models_bin/Bertweet/best_model")
    return bertweet, bertweet.tokenizer

def load_bertweet_multi():
    bertweet = BertweetModelMulticlass(num_labels=5, model_name="vinai/bertweet-base")
    bertweet.load_model("models/models_mul/Bertweet/best_model")
    return bertweet, bertweet.tokenizer
