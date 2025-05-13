import fasttext
import csv
from scripts.config import *

DATASET = f"{PWD}/data/interim"
TRAINING = f"{DATASET}/train"
VALIDATION = f"{DATASET}/val"
TEST = f"{DATASET}/test"

FAST_TEXT = f"{MODEL_DIR}/crawl-300d-2M-subword"
PRETRAINED_DIM=300 # When ValueError: Dimension of pretrained vectors (x) does not match dimension (y)! set this var to x
FINE_TUNED = f"{MODEL_DIR}/fine-tuned.bin"

EPOCHS = 15

class FineFastText:
    def __init__(self, model_path: str = FAST_TEXT) -> None:
        vec = model_path + ".vec"
        # bin = model_path + ".bin" #In theory used for zero-shot but the model is not a classifier

        assert os.path.exists(vec), model_path
        # assert os.path.exists(bin), model_path

        self.model_path = vec
        self.model = None #fasttext.load_model(bin)

    def fine_tune(self, epochs: int = EPOCHS):
        for i in range(epochs):
            print(f"Epoch {i}")
            # Train the model
            self.model = fasttext.train_supervised(input=f"{TRAINING}.txt", pretrainedVectors=self.model_path, dim=PRETRAINED_DIM)

            # Evaluate the model
            result = self.model.test(f"{VALIDATION}.txt")
            print(f'Epoch {i} validation accuracy: {result[1]} - f1 score {2*(result[1]*result[2]) / (result[1]+result[2])}')

        # Save the model
        self.model.save_model(FINE_TUNED)

    def predict(self, samples: list[str]) -> list:
        if self.model is None:
          self.model = fasttext.load_model(f"{FINE_TUNED}")

        results = []
        for sample in samples:
            results.append(self.model.predict(sample))
        return results

    @staticmethod
    def prepare_dataset():
        assert os.path.exists(f"{TRAINING}.csv")
        assert os.path.exists(f"{VALIDATION}.csv")
        assert os.path.exists(f"{TEST}.csv")

        for dataset in [TRAINING, VALIDATION, TEST]:
            FineFastText._write_txt(dataset)

    @staticmethod
    def _write_txt(dataset: str):
        if os.path.exists(f"{dataset}.txt"):
            # os.remove(f"{dataset}.txt")
            return
            
        txt = open(f"{dataset}.txt", "a", encoding="utf-8")
        file = open(f"{dataset}.csv", "r")

        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
          tweet = row['tweet_text']
          if "\n" in tweet:
            tweet = tweet.replace("\n", "\/n")
          txt.write(f"__label__{row['is_cyberbullying']} {tweet}\n")

        file.close()
        txt.close()

if __name__ == "__main__":
    print("Running")

    # dataset preparation
    FineFastText.prepare_dataset()
    print("Dataset ready")

    fast = FineFastText()

    # Fine tuning fasttext
    print("\nFine tuning")
    fast.fine_tune()
    tuned = fast.model.test(f"{TEST}.txt")

    print(f"Fine-tuned model binary accuracy: {tuned[1]}")
