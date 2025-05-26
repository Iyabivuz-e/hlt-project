import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, DataCollatorWithPadding
from peft import PeftModel
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    cohen_kappa_score,
    confusion_matrix,
    auc
)
from hf_mamba_classification import MambaForSequenceClassification

# CONFIGURATION
MODEL_NAME = "state-spaces/mamba-130m-hf"
ADAPTER_DIR = "mamba_base_lora/final_model"
BATCH_SIZE = 32
MAX_LENGTH = 128
RESULTS_DIR = "results"
DATA_DIR = Path("/content")  # adjust as needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_probs_from_hf_dataset(
    hf_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device
) -> np.ndarray:
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        hf_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=pin_memory
    )
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def save_results_and_plots(
    y_true: np.ndarray,
    probs: np.ndarray,
    set_name: str,
    threshold: float,
    out_dir: str = RESULTS_DIR
):
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Compute predictions
    y_pred = (probs >= threshold).astype(int)

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Collect metrics
    metrics = {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall (Sensitivity)": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "Negative Predictive Value": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, probs),
        "PR-AUC": average_precision_score(y_true, probs)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(out_dir, f"{set_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Save per-sample probabilities and predictions
    results_df = pd.DataFrame({
        "true_label": y_true,
        "probability": probs,
        "pred_label": y_pred
    })
    results_path = os.path.join(out_dir, f"{set_name}_predictions.csv")
    results_df.to_csv(results_path, index=False)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curve ({set_name})"
    )
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(out_dir, f"{set_name}_roc_curve.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"AP = {pr_auc:.3f}", linewidth=2)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision–Recall Curve ({set_name})"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    pr_path = os.path.join(out_dir, f"{set_name}_pr_curve.png")
    fig.savefig(pr_path, dpi=300)
    plt.close(fig)

    logger.info(f"[Saved] {set_name} metrics -> {metrics_path}")
    logger.info(f"[Saved] {set_name} predictions -> {results_path}")
    logger.info(f"[Saved] {set_name} ROC curve -> {roc_path}")
    logger.info(f"[Saved] {set_name} PR curve -> {pr_path}")


def main():
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = MambaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, use_cache=False
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device)

    # Load data
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv")
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")

    test_ds = Dataset.from_pandas(test_df)
    val_ds = Dataset.from_pandas(val_df)
    raw_dataset = DatasetDict({"test": test_ds, "val": val_ds})

    # Preprocessing function
    def preprocess(examples):
        texts = [str(x) for x in examples["text"]]
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        encodings["labels"] = [int(x) for x in examples["label"]]
        return encodings

    # Tokenize datasets
    dataset = raw_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["text", "label"]
    )

    # Extract splits
    y_val = np.array(dataset["val"]["labels"])
    y_test = np.array(dataset["test"]["labels"])

    # Compute validation probabilities and find optimal threshold
    probs_val = compute_probs_from_hf_dataset(dataset["val"], model, tokenizer, device)
    fpr, tpr, thresholds = roc_curve(y_val, probs_val)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_thresh = thresholds[best_idx]
    logger.info(f"Optimal threshold from validation: {best_thresh:.3f}")

    # Compute test probabilities
    probs_test = compute_probs_from_hf_dataset(dataset["test"], model, tokenizer, device)

    # Save results and plots
    save_results_and_plots(y_test, probs_test, set_name="test_youden", threshold=best_thresh)
    save_results_and_plots(y_test, probs_test, set_name="test_standard", threshold=0.5)

test()