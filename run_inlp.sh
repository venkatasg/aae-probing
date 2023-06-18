#!/bin/bash -x
#SBATCH -J aae-inlp
#SBATCH -o output-inlp.txt
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-user=venkatasg@utexas.edu
#SBATCH --mail-type=all


module reset

cd $WORK
source miniconda3/etc/profile.d/conda.sh
conda activate bertweet
cd aae/code/

for layer in {1..11} ;
    do
    for seed in 1 437 24 ;
        do
        python aae-collect-states.py --layer $layer --seed $seed --data_path ../data/dialect_data/dialect_with_preds.tsv --sampling_strat diff
        python aae-inlp.py --layer $layer --seed $seed
        done
    done
