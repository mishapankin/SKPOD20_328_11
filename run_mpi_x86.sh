#!/bin/bash

for n in {500..3001..500}
do
	for t in {1..6}
	do
        printf "%d %d " $n $t >> result_mpi_x86
		mpiexec -n $t ./mpi -r $n >> result_mpi_x86
	done
done
