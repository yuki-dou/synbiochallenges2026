import logging
import pandas as pd
from pathlib import Path
from model.saprot.saprot_foldseek_mutation_model import SaprotFoldseekMutationModel
from utils.foldseek_util import get_struc_seq


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
STEP_1_DIR = SCRIPT_DIR.parent / 'step1-create-candidates'
PDB_PATH = DATA_DIR / "2B3P_new.pdb"
FOLDSEEK_BIN = Path("bin/foldseek")
INPUT_CSV =  STEP_1_DIR / "500-candidates.csv"
OUTPUT_CSV = Path("results-saprot.csv")
MODEL_CONFIG_PATH = "SaProt_650M_PDB"
CHAIN = "A"


def load_structure(pdb_path: Path, foldseek_bin: Path, chain: str) -> str:
    """Parse PDB with Foldseek and return the combined (AA + structural) sequence."""
    parsed = get_struc_seq(
        str(foldseek_bin),
        str(pdb_path),
        [chain],
        plddt_mask=False
    )[chain]
    _, _, combined_seq = parsed
    logger.info("Combined sequence length: %d", len(combined_seq))
    return combined_seq


def load_model(config_path: str):
    """Initialize SaProtFoldseekMutationModel on CUDA."""
    config = {
        "foldseek_path": None,
        "config_path": config_path,
        "load_pretrained": True,
    }
    model = SaprotFoldseekMutationModel(**config)
    model.eval()
    model.to("cuda")
    return model


def main():
    combined_seq = load_structure(PDB_PATH, FOLDSEEK_BIN, CHAIN)
    model = load_model(MODEL_CONFIG_PATH)

    df = pd.read_csv(INPUT_CSV)
    logger.info("Scoring %d mutants with SaProt", len(df))

    scores = []
    for i, mutations in enumerate(df["Mutations"]):
        mut_info = mutations.replace(";", ":")
        score = model.predict_mut(combined_seq, mut_info)
        scores.append(score)
        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d", i + 1, len(df))

    df["saprot_score"] = scores
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    logger.info("Saved results to %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()