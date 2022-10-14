#!/bin/bash -x
#SBATCH -J aae-inlp
#SBATCH -o output-inlp.txt
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mail-user=venkatasg@utexas.edu
#SBATCH --mail-type=all


module reset

cd $WORK
source miniconda3/etc/profile.d/conda.sh
conda activate kyle
cd aae

for layer {0..11} ;
    do 
    for seed in 1 437 24;
        do
        python hate-speech-inlp.py --layer $layer --num_classifiers 32 --seed $seed
        done
    done
