#!/bin/bash
#SBATCH -N 1 -p gputest --gres=gpu:k80:1 -t 15 --mem=8G

module purge
module load cuda-env

make
srun ./main benchmark ../../input/benchmark.dat > ../../output/benchmark/k80.dat
