import fasttext
import gdown
import os

FASTTEXT_BINARY_ID = "1_iCbXY1yyZ536iNnzowJQFzJ9sSVvz4J"
FASTTEXT_MULTICLASS_ID = "1-E62aKmSF04OnA_8112XQwhFAUR2Uf3s"

def load_fasttext(mode: str):
    os.makedirs("models", exist_ok=True)
    output_path = f"models/fasttext_{mode}.bin"

    if not os.path.exists(output_path):
        file_id = FASTTEXT_MULTICLASS_ID if mode != "binary" else FASTTEXT_BINARY_ID
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    
    return fasttext.load_model(output_path)

if __name__ == "__main__":
    model = load_fasttext("binary")