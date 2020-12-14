#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stdbool.h>

void fill_matrix(double *matr, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matr[i * N + j] = rand() % 10;
        }
    }
}

int clamp(int x, int s) {
    return (x + s) % s;
}

double adabs(double x) {
    return (x > 0)? x: -x;
}

bool are_equal(double *A, double *B, int N, double eps) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (adabs(A[i * N + j] - B[i * N + j]) > eps) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    bool test = argc > 1 && strcmp("-t", argv[1]) == 0;

    int rank, pnum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);

    int N = 1000;
    if (argc > 2 && strcmp("-r", argv[1]) == 0) {
        sscanf(argv[2], "%d", &N);
    }
    int tape_size = (N - 1) / pnum + 1;

    double *A_tape = calloc(tape_size * N, sizeof(*A_tape));
    double *B_tape = calloc(tape_size * N, sizeof(*B_tape));
    double *C_tape = calloc(tape_size * N, sizeof(*C_tape));

    double *A, *B, *C, *E = NULL;
    double t1;
    if (rank == 0) {
        A = calloc(tape_size * pnum * N, sizeof(*A));
        B = calloc(tape_size * pnum * N, sizeof(*B));
        if (test) {
            E = calloc(tape_size * pnum * N, sizeof(*E));
        }
        fill_matrix(A, N);
        fill_matrix(B, N);

        if (test) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int k = 0; k < N; ++k) {
                        E[i * N + j] += A[i * N + k] * B[j * N + k]; 
                    }
                }
            }
        }

        C = calloc(tape_size * pnum * N, sizeof(*C));
        t1 = MPI_Wtime();
    }
    MPI_Scatter(A, tape_size * N, MPI_DOUBLE, A_tape, tape_size * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, tape_size * N, MPI_DOUBLE, B_tape, tape_size * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < tape_size; ++i) {
        for (int r = 0; r < pnum; ++r) {
            for (int j = 0; j < tape_size; ++j) {
                for (int k = 0; k < N; ++k) {
                    C_tape[i * N + j + clamp(r + rank, pnum) * tape_size] += A_tape[i * N + k] * B_tape[j * N + k];
                }
            }
            MPI_Status status;
            MPI_Sendrecv_replace(B_tape, tape_size * N, MPI_DOUBLE, clamp(rank - 1, pnum), 1, clamp(rank + 1, pnum), 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Gather(C_tape, tape_size * N, MPI_DOUBLE, C, tape_size * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double t2 = MPI_Wtime() - t1;
        printf("%.2f\n", t2);
        if (test) {
            if (are_equal(E, C, N, 1e-6)) {
                puts("\x001b[32mSUCCESS\x001b[0m");
            } else {
                puts("\x001b[31mFAIL\x001b[0m");
            }
        }

        free(A);
        free(B);
        free(C);
        if (E) {
            free(E);
        } 
    }

    free(A_tape);
    free(B_tape);
    free(C_tape);

    MPI_Finalize();
    return 0;
}