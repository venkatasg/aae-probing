#!/bin/bash -x
#SBATCH -J aae-alter
#SBATCH -o output-alter-sample-intervention.txt
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

for seed in 1 437 24 ;
    do
    # Pre intervention values
    python aae-alter.py --layer 11 --seed 1 --num_classifiers 0 --alpha 0 --data_path ../data/dialect_data/dialect_with_preds.tsv --reps_path reps_random
    for layer in {1..11} ;
        do
        for alpha in 4 -4 ;
            do
            for  dir in 8 32 ;
                do
                for reps_path in reps_diff ;
                    do
                        python aae-alter.py --layer $layer --seed $seed --num_classifiers $dir --alpha $alpha --data_path ../data/dialect_data/dialect_with_preds.tsv --reps_path $reps_path
                        python aae-alter.py --layer $layer --seed $seed --num_classifiers $dir --alpha $alpha --control --data_path ../data/dialect_data/dialect_with_preds.tsv --reps_path $reps_path
                    done
                done
            done
        done
    done
