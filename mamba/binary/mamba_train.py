from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
from mamba_ssm import selective_scan_fn
from google.colab import drive
from peft import LoraConfig, get_peft_model, TaskType
from hf_mamba_classification import MambaForSequenceClassification
import pandas as pd
from datasets import Dataset, DatasetDict
import os
import evaluate
import glob
import inspect, os
import math
import torch
os.environ["HF_DATASETS_CACHE"] = "/content/hf_cache"
MODEL_NAME = "state-spaces/mamba-130m-hf"
NUM_LABELS = 2
TRAIN_CSV = "/content/train_clean.csv"
VAL_CSV   = "/content/val_clean.csv"
OUTPUT_DIR = "mamba_base_lora"
max_length = 128
BATCH_SIZE = 32
NUM_EPOCHS = 18
LR = 2e-4
LORA_R = 12
LORA_ALPHA = 32
LORA_DROP = 0.05
def train():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = MambaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = NUM_LABELS, use_cache = False, id2label = id2label, label2id = label2id)
    model.to("cuda")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)

    raw_dataset = DatasetDict({"train": train_ds, "validation": val_ds})
    torch.backends.cudnn.benchmark = True

    def preprocess(examples):
        texts = [str(x) for x in examples["text"]]
        enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = [int(x) for x in examples["label"]]
        return enc

    dataset = raw_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["text", "label"],
    )

    peft_config = LoraConfig(
        task_type = TaskType.SEQ_CLS,
        target_modules = ["in_proj", "out_proj", "x_proj", "proj_in", "proj_out"],
        r = LORA_R,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROP,
        bias = 'none'
    )

    final_model = get_peft_model(model, peft_config)
    final_model.to("cuda")
    print(" % OF TRAINING")
    final_model.print_trainable_parameters()

    metric_acc = evaluate.load("accuracy")
    metric_f1  = evaluate.load("f1")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")

    def compute_metrics(p):
      preds = p.predictions.argmax(-1)
      return {
          "accuracy": metric_acc.compute(predictions = preds, references = p.label_ids)["accuracy"],
          "f1":       metric_f1.compute(predictions = preds, references = p.label_ids, average = "binary")["f1"],
          "precision":       metric_precision.compute(predictions = preds, references = p.label_ids, average="binary")["precision"],
          "recall":       metric_recall.compute(predictions = preds, references = p.label_ids, average="binary")["recall"],
      }

    final_model.gradient_checkpointing_enable()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32  = True
    drive.mount('/content/drive')
    OUTPUT_DIR_DRIVE = "/content/drive/MyDrive/mamba_checkpoints"
    import os
    os.makedirs(OUTPUT_DIR_DRIVE, exist_ok=True)
    #final_model = torch.compile(final_model, mode="default", fullgraph=False)
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR_DRIVE,
        per_device_train_batch_size = BATCH_SIZE,
        learning_rate               = LR,
        gradient_accumulation_steps = 8,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        dataloader_num_workers      = 2,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = "cosine",
        dataloader_pin_memory       = True,
        bf16                        = True,
        optim                       = "adamw_torch_fused",
        max_grad_norm               = 1.0,
        fp16                        = False,
        num_train_epochs            = NUM_EPOCHS,
        logging_strategy            = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        remove_unused_columns       = False,
        greater_is_better           = True,
        report_to                   = "none",
        label_names                 = ["labels"]
    )

    #final_model = torch.compile(final_model)
    with torch.no_grad():
      torch.nn.init.kaiming_uniform_(final_model.classifier.weight, a = math.sqrt(5))

    trainer = Trainer(
        model               = final_model,
        args                = training_args,
        train_dataset       = dataset["train"],
        tokenizer           = tokenizer,
        eval_dataset        = dataset["validation"],
        data_collator       = DataCollatorWithPadding(tokenizer, return_tensors = 'pt'),
        compute_metrics     = compute_metrics
    )

    all_ckpts = sorted(
    glob.glob(os.path.join(OUTPUT_DIR_DRIVE, "checkpoint-*")),
    key=lambda x: int(x.split("-")[-1])
    )
    if all_ckpts:
        print("🔄 Riprendo da:", all_ckpts[-1])
        trainer.train(resume_from_checkpoint=all_ckpts[-1])
    else:
        print("🔄 Nessun checkpoint trovato, inizio da zero")
        trainer.train()
    metrics = trainer.evaluate()
    print("Final evaluation:", metrics)

    trainer.save_model(os.path.join(OUTPUT_DIR_DRIVE, "final_model"))

train()