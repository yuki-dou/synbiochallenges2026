#!/bin/bash
./foldx_20270131 -c BuildModel --pdb 2B3P_new.pdb --mutant-file individual_list.txt --numberOfRuns 5 | tee mutations.log