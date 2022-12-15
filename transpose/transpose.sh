#!/bin/bash -l

#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=short-hsw
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=24
#SBATCH --output=transpose.out

module load gcc/9.2.0
module load openmpi/3.1.4

mpicc -o transpose main.c -lm -Wall -Wextra -g -O3

echo "STRONG"

EXP=13

for i in 1 2 4 8 12 16 20 24
do
    mpiexec -n $i transpose $EXP
done


echo
echo "WEAK"
EXP=13
for i in 1 4 16
do
    mpiexec -n $i transpose $EXP
    ((EXP+=1))
done
