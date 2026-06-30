import csv

with open('500-candidates.csv', mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    data_list = list(reader)

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

sequences = []
for seq in data_list:
    seq_processed = seq['Sequence']
    sequence = SeqRecord(
        Seq(seq_processed),
        id=seq['Mutations'],
        description=f"muts-{seq['Num_mutations']}"
    )
    sequences.append(sequence)

with open('500-candidates.fasta', "w") as f:
    SeqIO.write(sequences, f, "fasta")
