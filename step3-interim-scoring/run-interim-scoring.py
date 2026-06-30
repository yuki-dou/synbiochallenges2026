import logging
import pandas as pd
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MPNN_FILE = PROJECT_DIR / "step2.1-proteinmpnn" / "results-proteinmpnn.csv"
ESM_FILE = PROJECT_DIR / "step2.2-esm" / "results-esm.csv"
SAPROT_FILE = PROJECT_DIR / "step2.3-saprot" / "results-saprot.csv"

OUTPUT_FILE = Path("top-50-candidates.csv")
TOP_N = 50

WEIGHTS = {
    "MPNN": 0.60,
    "SaProt": 0.25,
    "ESM": 0.15,
}

MPNN_INTERNAL = {"base": 0.6, "stability": 0.4}


def load_all_scores() -> pd.DataFrame:
    """Load MPNN, SaProt and ESM scores, merge on Sequence."""
    mpnn = pd.read_csv(MPNN_FILE)
    saprot = pd.read_csv(SAPROT_FILE)
    esm = pd.read_csv(ESM_FILE)

    if "sequence" in mpnn.columns:
        mpnn = mpnn.rename(columns={"sequence": "Sequence"})

    merged = (
        mpnn
        .merge(saprot[["Sequence", "saprot_score"]], on="Sequence", how="inner")
        .merge(esm[["Sequence", "esm_score"]], on="Sequence", how="inner")
    )
    logger.info("Merged %d candidates", len(merged))
    return merged


def percentile_rank(series: pd.Series, reverse: bool = False) -> pd.Series:
    """Percentile rank normalization. Set reverse=True when lower is better."""
    rank = series.rank(pct=True)
    return 1.0 - rank if reverse else rank


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized score columns for each model."""
    df["mpnn_base_norm"] = percentile_rank(df["base_score"], reverse=True)
    df["mpnn_stability_norm"] = percentile_rank(df["stability_score"], reverse=True)

    df["MPNN_score"] = (
        MPNN_INTERNAL["base"] * df["mpnn_base_norm"] +
        MPNN_INTERNAL["stability"] * df["mpnn_stability_norm"]
    )
    df["SaProt_score"] = percentile_rank(df["saprot_score"], reverse=False)
    df["ESM_score"] = percentile_rank(df["esm_score"], reverse=True)
    return df


def ensemble_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted ensemble score and rank candidates."""
    df["Ensemble_score"] = (
        WEIGHTS["MPNN"] * df["MPNN_score"] +
        WEIGHTS["SaProt"] * df["SaProt_score"] +
        WEIGHTS["ESM"] * df["ESM_score"]
    )
    return df.sort_values("Ensemble_score", ascending=False)


def select_top(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Keep top N candidates and assign sequential IDs."""
    top = df.head(n).copy()
    top.insert(0, "candidate_id", range(1, n + 1))
    return top


def main():
    df = load_all_scores()
    df = normalize_scores(df)
    df = ensemble_score(df)
    top50 = select_top(df)

    top50.to_csv(OUTPUT_FILE, index=False)
    logger.info("Saved %d candidates to %s", len(top50), OUTPUT_FILE)

    report_cols = [
        "candidate_id", "Group", "Mutations",
        "MPNN_score", "SaProt_score", "ESM_score", "Ensemble_score"
    ]
    present = [c for c in report_cols if c in top50.columns]
    print(top50[present].to_string(index=False))


if __name__ == "__main__":
    main()