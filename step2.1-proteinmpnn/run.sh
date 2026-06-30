#!/bin/bash

mkdir -p logs

echo "Processing with noise 0.00"
python protein_mpnn_run.py --pdb_path $1 --path_to_fasta $2 --score_only 1 --out_folder ./scoring_noise_0.00 --batch_size 1 | tee logs/noise_0.00.log 2>&1

echo "Processing with noise 0.05"
python protein_mpnn_run.py --pdb_path $1 --path_to_fasta $2 --score_only 1 --backbone_noise 0.05 --out_folder ./scoring_noise_0.05 --batch_size 1 | tee logs/noise_0.05.log 2>&1

echo "Processing with noise 0.10"
python protein_mpnn_run.py --pdb_path $1 --path_to_fasta $2 --score_only 1 --backbone_noise 0.10 --out_folder ./scoring_noise_0.10 --batch_size 1 | tee logs/noise_0.10.log 2>&1

echo "Processing with noise 0.15"
python protein_mpnn_run.py --pdb_path $1 --path_to_fasta $2 --score_only 1 --backbone_noise 0.15 --out_folder ./scoring_noise_0.15 --batch_size 1 | tee logs/noise_0.15.log 2>&1

echo "Processing with noise 0.20"
python protein_mpnn_run.py --pdb_path $1 --path_to_fasta $2 --score_only 1 --backbone_noise 0.20 --out_folder ./scoring_noise_0.20 --batch_size 1 | tee logs/noise_0.20.log 2>&1

echo "DONE"
