import pandas as pd
import random
import re
import logging
from Bio import Align
from pathlib import Path


SEED = 42
random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Mutation constraints
FORBIDDEN_POSITIONS = {1, 65, 66, 67}
CAUTION_POSITIONS = {222, 87, 148, 203}
HYDROPHOBIC_CORE = {
    46, 54, 57, 71, 73, 80, 101, 112, 114,
    145, 151, 163, 170, 188, 192, 201, 221, 231
}
HYDROPHOBIC_ALLOWED = set('ILVFWMA')
POSITIVE_ONLY = {87: set('KR')}

MAX_TRIMMED_POS = 229 # Length of the sequence from PDB
MAX_ATTEMPTS = 200000
TARGET_PER_GROUP = 100

GROUP_RANGES = {
    '1-10': range(1, 11),
    '11-20': range(11, 21),
    '21-30': range(21, 31),
    '31-40': range(31, 41),
    '41-50': range(41, 51)
}


def load_fasta(path: Path) -> dict:
    """Parse a FASTA file, returns {header: sequence}."""
    seqs, name, seq = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('>'):
                if name:
                    seqs[name] = ''.join(seq).upper()
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if name:
            seqs[name] = ''.join(seq).upper()
    return seqs


def build_position_map(seq_a: str, seq_b: str) -> dict:
    """Map residue positions between two sequences via global alignment."""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligned = aligner.align(seq_a, seq_b)[0]
    mapping, p1, p2 = {}, 0, 0
    for x, y in zip(aligned[0], aligned[1]):
        if x != '-':
            p1 += 1
        if y != '-':
            p2 += 1
        if x != '-' and y != '-':
            mapping[p1] = p2
    return mapping


def load_references() -> tuple:
    """Load sfGFP and avGFP from the requirement FASTA."""
    fasta_path = Path('requirement') / 'AAseqs of 5 GFP proteins_20260511.txt'
    refs = load_fasta(fasta_path)
    ref_sfgfp = refs['sfGFP']
    ref_avgfp = refs['avGFP']
    logger.info("avGFP length: %d", len(ref_avgfp))
    logger.info("sfGFP length: %d", len(ref_sfgfp))
    return ref_sfgfp, ref_avgfp


def build_mutation_pool(ref_sfgfp: str, pos_map: dict) -> dict:
    """
    Build a dictionary of allowed mutations per sfGFP position
    using the avGFP mutational landscape.
    """
    csv_path = Path('requirement') / 'GFP_data.csv'
    df = pd.read_csv(csv_path)
    df_av = df[
    df['GFP type'].str.contains('avGFP', case=False, na=False)
    ].sort_values('Brightness', ascending=False)

    mutation_dict = {}
    for _, row in df_av.iterrows():
        muts = str(row['aaMutations'])
        if muts.upper() == 'WT':
            continue
        for mut in muts.split(':'):
            if len(mut) < 3:
                continue
            try:
                old, pos, new = mut[0], int(mut[1:-1]), mut[-1]
                if pos not in pos_map:
                    continue
                sf_pos = pos_map[pos]
                if sf_pos in FORBIDDEN_POSITIONS:
                    continue
                if ref_sfgfp[sf_pos - 1] == new:
                    continue
                if sf_pos in HYDROPHOBIC_CORE and new not in HYDROPHOBIC_ALLOWED:
                    continue
                if sf_pos in POSITIVE_ONLY and new not in POSITIVE_ONLY[sf_pos]:
                    continue
                mutation_dict.setdefault(sf_pos, set()).add(new)
            except (ValueError, IndexError):
                continue
    return mutation_dict


def filter_for_trimmed(mutation_dict: dict) -> dict:
    """
    Keep only positions that survive the trimmed-sequence representation:
    remove M1 (pos 1), replace CRO (TYG) 65-67 with X, shift indices.
    """
    filtered = {}
    for pos, aas in mutation_dict.items():
        if pos == 1 or 65 <= pos <= 67:
            continue
        if 2 <= pos <= 64:
            new_pos = pos - 1
        elif pos >= 68:
            new_pos = pos - 3
        else:
            continue
        if new_pos <= MAX_TRIMMED_POS:
            filtered[pos] = aas
    return {k: list(v) for k, v in filtered.items()}


def clean_sequence(raw: str) -> str:
    """Strip comments, whitespace; uppercase."""
    return str(raw).split('#')[0].replace(' ', '').upper().strip()


def load_exclusion_set() -> set:
    """Load sequences that must be avoided (exact or ≥99% identity)."""
    excl_path = Path('requirement') / 'Exclusion_List.csv'
    excl = pd.read_csv(excl_path)
    exclusion_set = set()
    for s in excl['Sequence'].dropna():
        seq = clean_sequence(s)
        if 220 <= len(seq) <= 250:
            exclusion_set.add(seq)
    logger.info("Exclusion set size: %d", len(exclusion_set))
    return exclusion_set


def sequence_identity(seq_a: str, seq_b: str) -> float:
    """Fraction of identical residues for equal-length sequences."""
    if len(seq_a) != len(seq_b):
        return 0.0
    return sum(x == y for x, y in zip(seq_a, seq_b)) / len(seq_a)


def check_exclusion(seq: str, exclusion_set: set) -> tuple:
    """
    Return (allowed: bool, near_exclusion: bool).
    `near_exclusion` is True when ≥99% identity to an excluded sequence.
    """
    if seq in exclusion_set:
        return False, True
    for ex_seq in exclusion_set:
        if sequence_identity(seq, ex_seq) >= 0.99:
            return True, True
    return True, False


def generate_candidate(template: str, num_mutations: int, mutation_dict: dict) -> tuple:
    """
    Introduce `num_mutations` random mutations into `template`.
    Returns (sequence, mutation_string, risk_score, critical_string)
    or (None, None, 0, None) on failure.
    """
    available = list(mutation_dict.keys())
    if num_mutations > len(available):
        num_mutations = len(available)
    if num_mutations == 0:
        return None, None, 0, None

    seq = list(template)
    chosen = random.sample(available, num_mutations)
    info, risk, critical = [], 0, []

    for pos in sorted(chosen):
        new_aa = random.choice(mutation_dict[pos])
        old_aa = seq[pos - 1]
        seq[pos - 1] = new_aa
        token = f'{old_aa}{pos}{new_aa}'
        info.append(token)
        if pos in CAUTION_POSITIONS:
            risk += 1
            critical.append(f'{token}_CAUTION')
        if pos in HYDROPHOBIC_CORE:
            risk += 2
            critical.append(f'{token}_CORE')

    return ''.join(seq), ';'.join(info), risk, ';'.join(critical)


def generate_all_candidates(ref_sfgfp: str, mutation_dict: dict, exclusion_set: set) -> list:
    """Generate 500 diverse candidates across 5 mutational-load groups."""
    all_candidates = []
    seen = set()

    for group_name, load_range in GROUP_RANGES.items():
        count = attempts = 0
        logger.info("Generating group %s", group_name)
        while count < TARGET_PER_GROUP and attempts < MAX_ATTEMPTS:
            attempts += 1
            n = random.choice(list(load_range))
            if n > len(mutation_dict):
                if len(mutation_dict) == 0:
                    break
                continue
            seq, muts, risk, critical = generate_candidate(ref_sfgfp, n, mutation_dict)
            if seq is None:
                continue
            if not seq.startswith('M') or not (220 <= len(seq) <= 250):
                continue
            if seq in seen:
                continue
            allowed, near_excl = check_exclusion(seq, exclusion_set)
            if not allowed:
                continue
            seen.add(seq)
            all_candidates.append({
                'Group': group_name,
                'Num_mutations': n,
                'Mutations': muts,
                'Risk_score': risk,
                'Critical_mutations': critical,
                'Near_exclusion': near_excl,
                'Sequence': seq
            })
            count += 1
        logger.info("%s → %d candidates", group_name, count)
    return all_candidates


def shift_mutation_token(token: str, max_pos: int = MAX_TRIMMED_POS) -> str | None:
    """
    Convert a mutation from full-length sfGFP indexing to trimmed indexing.
    Trimmed sequence omits M1 and replaces TYG(65-67) with X.
    """
    match = re.match(r'([A-Z])(\d+)([A-Z])(_.*)?$', token)
    if not match:
        return None
    old, pos, new, suffix = match.groups()
    pos = int(pos)
    if pos == 1:
        return None
    if 2 <= pos <= 64:
        new_pos = pos - 1
    elif pos >= 68:
        new_pos = pos - 3
    else:
        return None
    if new_pos > max_pos:
        return None
    return f'{old}{new_pos}{new}{suffix or ""}'


def shift_mutations(mut_str: str, max_pos: int = MAX_TRIMMED_POS) -> str:
    """Apply `shift_mutation_token` to every mutation in the string."""
    if not mut_str:
        return ''
    parts = []
    for token in mut_str.split(';'):
        shifted = shift_mutation_token(token, max_pos)
        if shifted:
            parts.append(shifted)
    return ';'.join(parts)


def trim_sequence(seq: str) -> str:
    """Remove M1, replace 65-67 (0-based 64-66) with 'X'."""
    return seq[1:64] + 'X' + seq[67:232]


def process_candidates(candidates: list) -> pd.DataFrame:
    """Apply index shifting and sequence trimming to all candidates."""
    processed = []
    for cand in candidates:
        trimmed_seq = trim_sequence(cand['Sequence'])
        new_muts = shift_mutations(cand['Mutations'])
        new_crit = shift_mutations(cand['Critical_mutations'])
        if not new_muts:
            continue
        processed.append({
            'Group': cand['Group'],
            'Num_mutations': len(new_muts.split(';')),
            'Mutations': new_muts,
            'Risk_score': cand['Risk_score'],
            'Critical_mutations': new_crit,
            'Near_exclusion': cand['Near_exclusion'],
            'Sequence': trimmed_seq
        })
    return pd.DataFrame(processed)


def main():
    ref_sfgfp, ref_avgfp = load_references()
    pos_map = build_position_map(ref_avgfp, ref_sfgfp)
    logger.info("Mapped positions: %d", len(pos_map))

    mutation_dict = build_mutation_pool(ref_sfgfp, pos_map)
    mutation_dict = filter_for_trimmed(mutation_dict)
    logger.info("Allowed positions after trimming: %d", len(mutation_dict))

    exclusion_set = load_exclusion_set()

    candidates = generate_all_candidates(ref_sfgfp, mutation_dict, exclusion_set)
    logger.info("Raw candidates: %d", len(candidates))

    df_processed = process_candidates(candidates)
    df_processed.to_csv('500-candidates.csv', index=False)
    logger.info("Saved 500-candidates.csv — %d rows", len(df_processed))


if __name__ == '__main__':
    main()