import numpy as np
import pandas as pd
import logging
from pathlib import Path
from Bio import SeqIO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
STEP_1_DIR = SCRIPT_DIR.parent / 'step1-create-candidates'

FASTA_PATH = STEP_1_DIR / "500-candidates.fasta"
CSV_PATH = STEP_1_DIR / "500-candidates.csv"
OUTPUT_PATH = Path("results-proteinmpnn.csv")

NOISE_FOLDERS = {
    0.00: Path("./scoring_noise_0.00"),
    0.10: Path("./scoring_noise_0.10"),
    0.15: Path("./scoring_noise_0.15"),
    0.20: Path("./scoring_noise_0.20"),
}

STABILITY_WEIGHTS = {0.10: 0.2, 0.15: 0.3, 0.20: 0.5}
FINAL_WEIGHTS = {"base": 0.45, "stability": 0.25, "noise20": 0.20, "risk": 0.10}


def load_scores(folder: Path, fasta_path: Path, noise_level: float) -> pd.DataFrame:
    """Parse FASTA and corresponding .npz score files from one noise folder."""
    ids, seqs = [], []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        ids.append(record.id)
        seqs.append(str(record.seq).upper())

    score_dir = folder / "score_only"
    npz_files = sorted(score_dir.glob("*.npz"))

    data = []
    for i, npz_file in enumerate(npz_files[:len(ids)]):
        scores = np.load(npz_file)
        data.append({
            "ID": ids[i],
            "sequence": seqs[i],
            "noise": noise_level,
            "mpnn_score": float(scores["score"].mean())
        })
    return pd.DataFrame(data)


def load_all_scores(fasta_path: Path) -> pd.DataFrame:
    """Load and concatenate scores from all noise levels."""
    all_data = []
    for noise_level, folder in NOISE_FOLDERS.items():
        if folder.exists():
            logger.info("Loading noise=%.2f from %s", noise_level, folder)
            all_data.append(load_scores(folder, fasta_path, noise_level))
        else:
            logger.warning("Folder %s not found, skipping noise=%.2f", folder, noise_level)
    return pd.concat(all_data, ignore_index=True)


def compute_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Pivot noise levels and compute base, stability and noise20 scores."""
    pivoted = scored.pivot_table(
        index=["ID", "sequence"],
        columns="noise",
        values="mpnn_score"
    ).reset_index()

    base = pivoted[0.00]
    stability = sum(
        weight * (pivoted[noise] - base)
        for noise, weight in STABILITY_WEIGHTS.items()
    )

    result = pivoted[["ID", "sequence"]].copy()
    result["base_score"] = base
    result["stability_score"] = stability
    result["noise20_score"] = pivoted[0.20]
    return result


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalization."""
    min_val, max_val = series.min(), series.max()
    return (series - min_val) / (max_val - min_val + 1e-9)


def compute_norms(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns for base, stability and noise20."""
    metrics["base_norm"] = normalize(metrics["base_score"])
    metrics["stab_norm"] = normalize(metrics["stability_score"])
    metrics["noise_norm"] = normalize(metrics["noise20_score"])
    return metrics


def merge_with_original(metrics: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Merge computed metrics into the original 500-candidates.csv."""
    original = pd.read_csv(csv_path).rename(columns={"Sequence": "sequence"})
    merged = original.merge(
        metrics[[
            "sequence", "base_score", "stability_score", "noise20_score",
            "base_norm", "stab_norm", "noise_norm"
        ]],
        on="sequence",
        how="left"
    )
    return merged


def compute_final_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk normalization and compute weighted final score."""
    partial = (
        FINAL_WEIGHTS["base"] * df["base_norm"] +
        FINAL_WEIGHTS["stability"] * df["stab_norm"] +
        FINAL_WEIGHTS["noise20"] * df["noise_norm"]
    )
    if "Risk_score" in df.columns:
        risk_norm = normalize(df["Risk_score"].fillna(0))
    else:
        risk_norm = 0.0
    df["final_score"] = partial + FINAL_WEIGHTS["risk"] * risk_norm
    return df


def main():
    # Load scores from all noise levels
    scored_all = load_all_scores(FASTA_PATH)
    logger.info("Total scored sequences: %d", len(scored_all))

    # Compute metrics and norms
    metrics = compute_metrics(scored_all)
    metrics = compute_norms(metrics)

    # Merge with original candidates and compute final score
    result = merge_with_original(metrics, CSV_PATH)
    result = compute_final_score(result)

    # Sort by final_score (lower is better) and save
    result = result.sort_values("final_score").reset_index(drop=True)
    result.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(result), OUTPUT_PATH)

    # Summary
    logger.info("Top 10 by final_score (lower is better):")
    top_cols = ["sequence", "Group", "Num_mutations", "final_score", "Risk_score"]
    for col in top_cols:
        if col not in result.columns:
            logger.warning("Column '%s' missing in final output", col)
    present_cols = [c for c in top_cols if c in result.columns]
    print(result[present_cols].head(10))

    logger.info("Group distribution:")
    if "Group" in result.columns:
        print(result["Group"].value_counts())


if __name__ == '__main__':
    main()