import logging
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

SCRIPT_DIR = Path(__file__).resolve().parent
STEP_1_DIR = SCRIPT_DIR.parent / 'step1-create-candidates'
INPUT_CSV = STEP_1_DIR / "500-candidates.csv"
OUTPUT_CSV = Path("results-esm.csv")


def load_model(model_name: str) -> tuple:
    """Load ESM-2 model and tokenizer, set to evaluation mode."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def score_sequence(sequence: str, model, tokenizer) -> float:
    """
    Compute cross-entropy loss for a single sequence.
    Lower values indicate higher naturalness under the model.
    """
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


def main():
    model, tokenizer = load_model(MODEL_NAME)
    df = pd.read_csv(INPUT_CSV)
    sequences = df["Sequence"].tolist()
    logger.info("Scoring %d sequences with %s", len(sequences), MODEL_NAME)

    scores = []
    for i, seq in enumerate(sequences):
        loss = score_sequence(seq, model, tokenizer)
        scores.append(loss)
        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d", i + 1, len(sequences))

    df["esm_score"] = scores
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved results to %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()