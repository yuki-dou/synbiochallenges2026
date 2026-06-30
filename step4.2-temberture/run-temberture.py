import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from temBERTure import TemBERTure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR.parent / "step3-interim-scoring" / "top-50-candidates.csv"
OUTPUT_PATH = Path("results-temberture.csv")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d candidates from %s", len(df), path)

    if "Sequence" not in df.columns:
        raise ValueError("Missing 'Sequence' column in input CSV")

    return df


def initialize_models() -> tuple:
    logger.info("Initializing models...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Using device: %s", device)

    logger.info("Loading TemBERTureCLS...")
    model_cls = TemBERTure(
        adapter_path='./temBERTure_CLS/',
        device=device,
        batch_size=1,
        task='classification'
    )

    logger.info("Loading TemBERTureTM replicas...")
    model_tm_r1 = TemBERTure(
        adapter_path='./temBERTure_TM/replica1/',
        device=device,
        batch_size=16,
        task='regression'
    )
    model_tm_r2 = TemBERTure(
        adapter_path='./temBERTure_TM/replica2/',
        device=device,
        batch_size=16,
        task='regression'
    )
    model_tm_r3 = TemBERTure(
        adapter_path='./temBERTure_TM/replica3/',
        device=device,
        batch_size=16,
        task='regression'
    )

    logger.info("All models loaded successfully")
    return model_cls, model_tm_r1, model_tm_r2, model_tm_r3


def process_classification(sequences: list, model_cls) -> tuple:
    logger.info("Processing classification for %d sequences...", len(sequences))

    predictions = []
    scores = []

    for seq in sequences:
        result = model_cls.predict(seq)
        predictions.append(result[0][0])
        scores.append(float(result[1][0]))

    logger.info("CLS processing complete")
    return predictions, scores


def process_regression_batch(sequences: list, model, replica_name: str) -> list:
    logger.info("Processing regression - %s...", replica_name)

    batch_size = 16
    predictions = []
    n_batches = len(sequences) // batch_size + (1 if len(sequences) % batch_size != 0 else 0)

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]

        batch_predictions = model.predict(batch_sequences)
        predictions.extend(batch_predictions)

    logger.info("%s processing complete", replica_name)
    return predictions


def process_sequences(
        df: pd.DataFrame,
        model_cls,
        model_tm_r1,
        model_tm_r2,
        model_tm_r3
) -> pd.DataFrame:
    sequences = df['Sequence'].tolist()
    logger.info("Processing %d sequences...", len(sequences))

    cls_predictions, cls_scores = process_classification(sequences, model_cls)

    tm_replica1 = process_regression_batch(sequences, model_tm_r1, "replica1")
    tm_replica2 = process_regression_batch(sequences, model_tm_r2, "replica2")
    tm_replica3 = process_regression_batch(sequences, model_tm_r3, "replica3")

    df_result = df.copy()
    df_result['cls_prediction'] = cls_predictions
    df_result['cls_score'] = cls_scores
    df_result['tm_replica1'] = tm_replica1
    df_result['tm_replica2'] = tm_replica2
    df_result['tm_replica3'] = tm_replica3
    df_result['tm_mean'] = df_result[['tm_replica1', 'tm_replica2', 'tm_replica3']].mean(axis=1)
    df_result['tm_std'] = df_result[['tm_replica1', 'tm_replica2', 'tm_replica3']].std(axis=1)

    return df_result


def main():
    logger.info("Starting TemBERTure scoring pipeline")

    df = load_data(INPUT_PATH)

    model_cls, model_tm_r1, model_tm_r2, model_tm_r3 = initialize_models()

    df_result = process_sequences(
        df,
        model_cls,
        model_tm_r1,
        model_tm_r2,
        model_tm_r3
    )

    logger.info("Saving results to %s...", OUTPUT_PATH)
    df_result.to_csv(OUTPUT_PATH, index=False)

    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()