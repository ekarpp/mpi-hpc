#!/bin/bash -l

#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=200M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=short-hsw
#SBATCH --exclusive
#SBATCH --output=qs.out

module load gcc/9.2.0
module load openmpi/3.1.4

mpicc -o qs -O3 main.c

EXP=24

echo "STRONG"
for i in 1 2 4 8 12 16 20 24
do
    mpiexec -n $i ./qs $EXP
done

echo
echo "WEAK"

EXP=23

for i in 1 2 4 8 16
do
    mpiexec -n $i ./qs $EXP
    ((EXP+=1))
done
