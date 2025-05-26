# before python
# git clone https://github.com/getorca/mamba_for_sequence_classification.git
# rm -rf mamba_for_sequence_classification/requirements.txt
# touch mamba_for_sequence_classification/requirements.txt
# pip3 install -q ./mamba_for_sequence_classification
# pip3 install -U causal-conv1d
# pip3 install accelerate
# pip3 install peft
# pip3 install transformers torch
# pip3 install --no-build-isolation --no-cache-dir -U mamba-ssm
from transformers import AutoTokenizer, DataCollatorWithPadding
from hf_mamba_classification import MambaForSequenceClassification
from peft import PeftModel
from types import MethodType
import torch

MAMBA_BASE_MODEL_NAME = "state-spaces/mamba-130m-hf"
MAMBA_ADAPTER_DIR_BINARY = "binary/mamba_base_lora/final_model/"
MAMBA_ADAPTER_DIR_MULTI = "multi/mamba_base_lora/"
TOKENIZER_MAX_LENGTH = 128

device = "cuda"

def __get_mamba(type : str):
    if (type == "binary"):
        tokenizer_ = AutoTokenizer.from_pretrained(MAMBA_BASE_MODEL_NAME)
        base_model = MambaForSequenceClassification.from_pretrained(MAMBA_BASE_MODEL_NAME, num_labels = 2, use_cache = False)
        model = PeftModel.from_pretrained(base_model, MAMBA_ADAPTER_DIR_BINARY).to(device)
    if (type == "multi"):
        tokenizer_ = AutoTokenizer.from_pretrained(MAMBA_BASE_MODEL_NAME)
        base_model = MambaForSequenceClassification.from_pretrained(MAMBA_BASE_MODEL_NAME, num_labels = 5, use_cache = False)
        model = PeftModel.from_pretrained(base_model, MAMBA_ADAPTER_DIR_MULTI).to(device)
    tokenizer = lambda string: tokenizer_(str(string), truncation = True, padding = "max_length", max_length = TOKENIZER_MAX_LENGTH, return_tensors = "pt").to(device)
    model.eval()
    def predict(self, text):
        with torch.no_grad():
            inputs = tokenizer(text)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            return probs.squeeze().tolist()
    model.predict = MethodType(predict, model)
    return model

def get_mamba_binary():
    return __get_mamba("binary")

def get_mamba_multi():
    return  __get_mamba("multi")
    
def __mamba_test():
    m = get_mamba_binary()
    m1 = get_mamba_multi()
    while (True):
        seq = input("Insert sentence")
        probs = m.predict(seq)
        probs2 = m1.predict(seq)
        print(probs)
        print(probs2)