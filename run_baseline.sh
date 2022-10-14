#!/bin/bash -x
#SBATCH -J aae-baseline
#SBATCH -o output-baseline.txt
#SBATCH -p gtx
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

python hate-speech-baseline.py --seed 1
python hate-speech-baseline.py --seed 437
python hate-speech-baseline.py --seed 24