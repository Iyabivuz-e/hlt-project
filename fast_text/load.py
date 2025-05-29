import fasttext
import os


def load_fasttext(mode: str):
    path = f"/content/drive/MyDrive/HLT_artifacts_group_5/fine-tuned_models/fasttext-{mode}.bin"
    return fasttext.load_model(path)

if __name__ == "__main__":
    model = load_fasttext("binary")

