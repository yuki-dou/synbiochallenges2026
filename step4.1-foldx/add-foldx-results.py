import logging
import re
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "step3-interim-scoring" / "top-50-candidates.csv"
FXOUT_PATH = Path("Dif_2B3P_new.fxout")
OUTPUT_PATH = Path("results-foldx.csv")

FOLDX_COLUMNS = [
    "Pdb", "total_energy", "Backbone_Hbond", "Sidechain_Hbond",
    "Van_der_Waals", "Electrostatics", "Solvation_Polar",
    "Solvation_Hydrophobic", "Van_der_Waals_clashes",
    "entropy_sidechain", "entropy_mainchain", "sloop_entropy",
    "mloop_entropy", "cis_bond", "torsional_clash",
    "backbone_clash", "helix_dipole", "water_bridge", "disulfide",
    "electrostatic_kon", "partial_covalent_bonds",
    "energy_Ionisation", "Entropy_Complex",
]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d candidates from CSV", len(df))

    if "candidate_id" not in df.columns:
        raise ValueError("Missing 'candidate_id' column")

    return df


def parse_fxout(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("2B3P"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                values = [float(x) for x in parts[1:]]
            except (ValueError, IndexError):
                continue
            rows.append([parts[0]] + values)

    fx = pd.DataFrame(rows, columns=FOLDX_COLUMNS[: len(rows[0])])
    logger.info("Parsed %d FoldX structures", len(fx))
    return fx


def extract_candidate_id(pdb_name: str) -> tuple[int, int]:
    match = re.search(r"_(\d+)_(\d+)\.pdb", pdb_name)
    if not match:
        raise ValueError(f"Cannot parse FoldX name: {pdb_name}")
    return int(match.group(1)), int(match.group(2))


def average_replicates(fx: pd.DataFrame) -> pd.DataFrame:
    parsed = fx["Pdb"].apply(extract_candidate_id)
    fx["candidate_id"] = parsed.apply(lambda x: x[0])
    fx["replica"] = parsed.apply(lambda x: x[1])

    replica_counts = fx.groupby("candidate_id").size()
    if not (replica_counts == 5).all():
        bad = replica_counts[replica_counts != 5].to_dict()
        logger.warning("Some candidates lack 5 replicas: %s", bad)

    logger.info("Unique candidates in FoldX: %d", fx["candidate_id"].nunique())

    numeric_cols = [c for c in fx.columns if c not in ("Pdb", "candidate_id", "replica")]
    return fx.groupby("candidate_id")[numeric_cols].mean().reset_index()


def main():
    df = load_csv(CSV_PATH)
    fx = parse_fxout(FXOUT_PATH)
    fx_avg = average_replicates(fx)

    missing = set(df["candidate_id"]) - set(fx_avg["candidate_id"])
    if missing:
        logger.warning("Missing FoldX data for candidates: %s", missing)

    df = df.merge(fx_avg, on="candidate_id", how="left")
    logger.info("After merge: %d candidates", len(df))

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved results to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()