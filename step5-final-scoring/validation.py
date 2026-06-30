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
PROJECT_DIR = SCRIPT_DIR.parent / 'step1-create-candidates' / 'requirement'
REF_SEQS = PROJECT_DIR / "AAseqs of 5 GFP proteins_20260511.txt"
CSV_PATH = Path("top-10-final-score.csv")
OUTPUT_CSV = Path("final-top10-restored.csv")
OUTPUT_FASTA = Path("final-top10-restored.fasta")
REF_NAME = "sfGFP"

TRANSITION_POS = 64


def load_fasta(path: Path) -> dict:
    seqs, name, seq = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(seq).upper()
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name:
            seqs[name] = "".join(seq).upper()
    return seqs


def parse_mutation(token: str) -> tuple[str, int, str, str] | None:
    match = re.match(r"([A-Z])(\d+)([A-Z])(_.*)?$", token)
    if not match:
        return None
    old, pos, new, suffix = match.groups()
    return old, int(pos), new, suffix or ""


def shift_mutations_back(mut_str: str) -> tuple[str, list[dict]]:
    if not mut_str or pd.isna(mut_str):
        return "", []

    shifted, parsed = [], []
    for token in str(mut_str).split(";"):
        token = token.strip()
        if not token:
            continue
        result = parse_mutation(token)
        if result is None:
            logger.warning("Cannot parse token: %s", token)
            continue
        old, pos, new, suffix = result
        new_pos = pos + 1 if pos < TRANSITION_POS else pos + 3
        shifted.append(f"{old}{new_pos}{new}{suffix}")
        parsed.append({"old": old, "pos": new_pos, "new": new, "raw": token})

    return ";".join(shifted), parsed


def restore_sequence(trimmed_seq: str, ref_seq: str) -> str:
    trimmed = trimmed_seq.strip().upper()
    ref = ref_seq.strip().upper()

    seq_with_m = "M" + trimmed
    x_idx = trimmed.find("X")
    if x_idx == -1:
        logger.warning("No 'X' found in trimmed sequence; assuming position 64")
        x_idx = 64
    x_pos_full = x_idx + 1
    seq_fixed = seq_with_m[:x_pos_full] + "TYG" + seq_with_m[x_pos_full + 1:]
    if len(seq_fixed) < len(ref):
        seq_fixed += ref[len(seq_fixed):]
    return seq_fixed


def sanity_check(restored: str, parsed_muts: list[dict], ref: str, candidate_id: int) -> bool:
    check = list(ref)
    for mut in parsed_muts:
        pos = mut["pos"]
        if pos > len(check):
            logger.error("Candidate %d: position %d out of range", candidate_id, pos)
            return False
        if check[pos - 1] != mut["old"]:
            logger.error(
                "Candidate %d: mismatch at %d (expected %s, got %s)",
                candidate_id, pos, mut["old"], check[pos - 1]
            )
            return False
        check[pos - 1] = mut["new"]
    check_str = "".join(check)
    if check_str == restored:
        logger.info("Candidate %d: sanity check passed", candidate_id)
        return True
    for i, (c1, c2) in enumerate(zip(check_str, restored)):
        if c1 != c2:
            logger.error(
                "Candidate %d: mismatch at %d (expected %s, got %s)",
                candidate_id, i + 1, c2, c1
            )
            break
    return False


def build_fasta_header(row: pd.Series) -> str:
    header = f">candidate_{row['candidate_id']}"
    muts = row.get("Mutations_shifted", row.get("Mutations", ""))
    if not pd.isna(muts) and muts:
        header += f" | mutations={muts}"
    for col in ["final_score", "Ensemble_score", "foldx_score"]:
        if col in row and not pd.isna(row[col]):
            header += f" | {col}={row[col]:.4f}"
    if "total_energy" in row and not pd.isna(row["total_energy"]):
        header += f" | total_energy={row['total_energy']:.2f}"
    return header


def main():
    refs = load_fasta(REF_SEQS)
    ref_seq = refs[REF_NAME]
    logger.info("Reference %s length: %d", REF_NAME, len(ref_seq))

    df = pd.read_csv(CSV_PATH)
    for col in ["candidate_id", "Mutations", "Sequence"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Available: {list(df.columns)}")
    logger.info("Loaded %d candidates from %s", len(df), CSV_PATH)

    foldx_cols = [c for c in df.columns if c.endswith("_foldx")]
    if foldx_cols:
        logger.info("Removing duplicate columns: %s", foldx_cols)
        df = df.drop(columns=foldx_cols)

    passed = []
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        restored = restore_sequence(row["Sequence"], ref_seq)
        shifted_str, parsed = shift_mutations_back(row["Mutations"])

        if not sanity_check(restored, parsed, ref_seq, cid):
            continue

        row_dict = row.to_dict()
        row_dict["Sequence"] = restored
        row_dict["Mutations_shifted"] = shifted_str
        row_dict["Length_full"] = len(restored)
        passed.append(row_dict)

    logger.info("Passed: %d / %d", len(passed), len(df))
    if not passed:
        logger.error("No candidates passed sanity check — aborting")
        raise SystemExit(1)

    df_out = pd.DataFrame(passed)

    cols = list(df_out.columns)
    if "Sequence" in cols:
        cols.remove("Sequence")
    if "Mutations_shifted" in cols:
        insert_pos = cols.index("Mutations_shifted") + 1
    else:
        insert_pos = cols.index("Mutations") + 1 if "Mutations" in cols else 2
    cols.insert(insert_pos, "Sequence")
    df_out = df_out[cols]

    df_out.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved CSV with %d rows: %s", len(df_out), OUTPUT_CSV)
    logger.info("Columns in output: %s", list(df_out.columns))

    with open(OUTPUT_FASTA, "w") as fh:
        for _, row in df_out.iterrows():
            fh.write(build_fasta_header(row) + "\n")
            fh.write(row["Sequence"] + "\n")
    logger.info("Saved FASTA: %s", OUTPUT_FASTA)


if __name__ == "__main__":
    main()