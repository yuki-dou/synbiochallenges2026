import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TEMBERTURE_FILE = SCRIPT_DIR.parent / "step4.2-temberture" / "results-temberture.csv"
FOLDX_FILE = SCRIPT_DIR.parent / "step4.1-foldx" / "results-foldx.csv"
OUTPUT = Path("top-10-final-score.csv")

MODEL_WEIGHTS = {
    "mpnn": 0.30,
    "saprot": 0.10,
    "esm": 0.05,
    "temberture": 0.40,
    "foldx": 0.15
}

FOLDX_WEIGHTS = {
    "energy": 0.60,
    "clashes": 0.20,
    "entropy": 0.10,
    "electro": 0.05,
    "solvation": 0.05
}


def load_data() -> pd.DataFrame:
    temp = pd.read_csv(TEMBERTURE_FILE)
    foldx = pd.read_csv(FOLDX_FILE)

    logger.info("TemBERTure candidates: %d", len(temp))
    logger.info("FoldX candidates: %d", len(foldx))

    if "candidate_id" not in temp.columns:
        raise ValueError("candidate_id missing in TemBERTure file")

    if "candidate_id" not in foldx.columns:
        raise ValueError("candidate_id missing in FoldX file")

    df = temp.merge(foldx, on="candidate_id", how="inner", suffixes=("", "_foldx"))
    logger.info("Merged candidates: %d", len(df))

    return df


def percentile_norm(series: pd.Series, reverse: bool = False) -> pd.Series:
    rank = series.rank(pct=True)
    return 1 - rank if reverse else rank


def add_model_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["mpnn_norm"] = percentile_norm(df["MPNN_score"], reverse=False)
    df["saprot_norm"] = percentile_norm(df["SaProt_score"], reverse=False)
    df["esm_norm"] = percentile_norm(df["ESM_score"], reverse=True)
    df["tm_norm"] = percentile_norm(df["tm_mean"], reverse=False)
    df["tm_variability_norm"] = percentile_norm(df["tm_std"], reverse=True)
    df["temberture_score"] = 0.85 * df["tm_norm"] + 0.15 * df["tm_variability_norm"]

    return df


def add_foldx_score(df: pd.DataFrame) -> pd.DataFrame:
    df["foldx_energy"] = percentile_norm(df["total_energy"], reverse=True)
    df["foldx_clashes"] = percentile_norm(df["Van_der_Waals_clashes"], reverse=True)
    df["foldx_entropy"] = percentile_norm(df["entropy_sidechain"], reverse=True)
    df["foldx_electro"] = percentile_norm(df["Electrostatics"], reverse=True)
    df["foldx_solvation"] = percentile_norm(df["Solvation_Hydrophobic"], reverse=True)

    df["foldx_score"] = (
            FOLDX_WEIGHTS["energy"] * df["foldx_energy"] +
            FOLDX_WEIGHTS["clashes"] * df["foldx_clashes"] +
            FOLDX_WEIGHTS["entropy"] * df["foldx_entropy"] +
            FOLDX_WEIGHTS["electro"] * df["foldx_electro"] +
            FOLDX_WEIGHTS["solvation"] * df["foldx_solvation"]
    )

    return df


def calculate_final(df: pd.DataFrame) -> pd.DataFrame:
    df["final_score"] = (
            MODEL_WEIGHTS["mpnn"] * df["mpnn_norm"] +
            MODEL_WEIGHTS["saprot"] * df["saprot_norm"] +
            MODEL_WEIGHTS["esm"] * df["esm_norm"] +
            MODEL_WEIGHTS["temberture"] * df["temberture_score"] +
            MODEL_WEIGHTS["foldx"] * df["foldx_score"]
    )

    return df


def main():
    logger.info("Starting final scoring")

    df = load_data()
    df = add_model_scores(df)
    df = add_foldx_score(df)
    df = calculate_final(df)

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "final_rank", range(1, len(df) + 1))

    top10 = df.head(10)

    cols = [
        "final_rank",
        "candidate_id",
        "Group",
        "Mutations",
        "final_score",
        "mpnn_norm",
        "saprot_norm",
        "esm_norm",
        "tm_mean",
        "tm_std",
        "foldx_score",
        "total_energy"
    ]

    cols = [c for c in cols if c in top10.columns]

    print("\nFINAL TOP-10\n")
    print(top10[cols].to_string(index=False))

    top10.to_csv(OUTPUT, index=False)
    logger.info("Saved %s", OUTPUT)


if __name__ == "__main__":
    main()