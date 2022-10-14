#!/bin/bash -x
#SBATCH -J aae-alter
#SBATCH -o output-alter.txt
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

for alpha in 2 -2 0;
    do
    for layer {0..11} ;
        do 
        for seed in 1 437 24;
            do
            python hate-speech-alter.py --layer $layer  --seed $seed  --alpha $alpha --control
            python hate-speech-alter.py --layer $layer  --seed $seed  --alpha $alpha
            done
        done
    done