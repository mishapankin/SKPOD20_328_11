omp: main_omp.c
	gcc -O3 -flto -fopenmp main_omp.c -o omp -march=native -std=gnu11

omp_no_opt: main_omp.c
	gcc -O0 -fopenmp main_omp.c -o omp -std=gnu11

mpi: main_mpi.c
	mpicc -O3 -flto main_mpi.c -o mpi -march=native -std=gnu11

run_mpi: mpi
	mpiexec -n 4 ./mpi

clean:
	rm -f ./omp
	rm -f ./mpi

.PHONY: clean