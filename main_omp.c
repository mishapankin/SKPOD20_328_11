#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>


#define OK_STR "[\x001b[32m OK\x001b[0m ]"
#define FAILED_STR "[\x001b[31m FAILED\x001b[0m ]"

typedef __attribute__((aligned(64))) double aligned_double;

aligned_double *create_matrix(int N, int M) {
    return aligned_alloc(64, N * M * sizeof(aligned_double));
}

void fill_matrix(aligned_double *matr, int N, int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            matr[i * M + j] = rand() % 10;
        }
    }
}

void copy_transposed(aligned_double *src, aligned_double *dest, int N, int M) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            dest[j * N + i] = src[i * M + j];
        }
    }
}

double time_sec(struct timespec t1, struct timespec t2) {
    return (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
}

void print_time(const char *str, struct timespec t1, struct timespec t2) {
    printf("%s time: %gs\n", str, time_sec(t1, t2));
}

void multiply_matrix(aligned_double *A, aligned_double *B, aligned_double *C, int N, int M, int K) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < M; ++j) {
                C[i * K + j] += A[i * M + k] * B[k * M + j];
            }
        }
    }
}


void multiply_matrix_omp1(aligned_double *A, aligned_double *B, aligned_double *C, int N, int M, int K, int T) {
    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0;

            #pragma omp simd
            for (int k = 0; k < K; ++k) {
                sum += A[i * M + k] * B[k * M + j];
            }
            C[i * K + j] = sum;
        }
    }
}


void multiply_matrix_omp2(aligned_double *A, aligned_double *B, aligned_double *C, int N, int M, int K, int T) {
    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            #pragma omp simd
            for (int j = 0; j < M; ++j) {
                C[i * K + j] += A[i * M + k] * B[k * M + j];
            }
        }
    }
}


void multiply_matrix_tr(aligned_double *A, aligned_double *TB, aligned_double *C, int N, int M, int K) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * M + k] * TB[j * K + k];
            }
            C[i * K + j] = sum;
        }
    }
}


void multiply_matrix_omp1_tr(aligned_double *A, aligned_double *TB, aligned_double *C, int N, int M, int K, int T) {
    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            double sum = 0;
            #pragma omp simd
            for (int k = 0; k < K; ++k) {
                sum += A[i * M + k] * TB[j * K + k];
            }
            C[i * K + j] = sum;
        }
    }
}


void multiply_matrix_omp2_tr(aligned_double *A, aligned_double *TB, aligned_double *C, int N, int M, int K, int T) {
    #pragma omp parallel for num_threads(T)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < K; ++k) {
            #pragma omp simd
            for (int j = 0; j < M; ++j) {
                C[i * K + j] += A[i * M + k] * TB[j * K + k];
            }
        }
    }
}

void print_matrix(aligned_double *matr, int N, int M) {
    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < M; ++j) {
            printf("%lf ", matr[i * M + j]);
        }
        printf("\n");
    }
}

aligned_double adabs(aligned_double x) {
    return (x > 0)? x: -x;
}

bool are_equal(aligned_double *A, aligned_double *B, int N, int M, aligned_double eps) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            if (adabs(A[i * M + j] - B[i * M + j]) > eps) {
                return false;
            }
        }
    }
    return true;
}

int test(int N) {
    printf("N: %d\n", N);
    int T = omp_get_max_threads();

    aligned_double *A = create_matrix(N, N);
    aligned_double *B = create_matrix(N, N);
    aligned_double *TB = create_matrix(N, N);
    aligned_double *E = create_matrix(N, N);
    aligned_double *C = create_matrix(N, N);

    fill_matrix(A, N, N);
    fill_matrix(B, N, N);
    copy_transposed(B, TB, N, N);

    multiply_matrix(A, B, E, N, N, N);

    int passed = 0;

    multiply_matrix_tr(A, TB, C, N, N, N);
    printf("Single thread TRANSPOSED ");
    if (are_equal(C, E, N, N, 1e-6)) {
        ++passed;
        puts(OK_STR);
    } else {
        puts(FAILED_STR);
    }
    memset(C, 0, N * N * sizeof(*C));

    multiply_matrix_omp1(A, B, C, N, N, N, T);
    printf("OMP 1 ");
    if (are_equal(C, E, N, N, 1e-6)) {
        ++passed;
        puts(OK_STR);
    } else {
        puts(FAILED_STR);
    }
    memset(C, 0, N * N * sizeof(*C));

    multiply_matrix_omp2(A, B, C, N, N, N, T);
    printf("OMP 2 ");
    if (are_equal(C, E, N, N, 1e-6)) {
        ++passed;
        puts(OK_STR);
    } else {
        puts(FAILED_STR);
    }
    memset(C, 0, N * N * sizeof(*C));

    multiply_matrix_omp1_tr(A, TB, C, N, N, N, T);
    printf("OMP 1 TRANSPOSED ");
    if (are_equal(C, E, N, N, 1e-6)) {
        ++passed;
        puts(OK_STR);
    } else {
        puts(FAILED_STR);
    }
    memset(C, 0, N * N * sizeof(*C));

    multiply_matrix_omp2_tr(A, TB, C, N, N, N, T);
    printf("OMP 2 TRANSPOSED ");
    if (are_equal(C, E, N, N, 1e-6)) {
        ++passed;
        puts(OK_STR);
    } else {
        puts(FAILED_STR);
    }

    free(A);
    free(B);
    free(C);
    free(TB);
    free(E);

    return passed;
}

void compare_versions(int SZ) {
    int N = SZ, M = SZ, K = SZ;
    int T = omp_get_max_threads();
    printf("%d\n", T);

    aligned_double *A = create_matrix(N, M);
    aligned_double *B = create_matrix(M, K);
    aligned_double *TB = create_matrix(K, M);
    fill_matrix(A, N, M);
    fill_matrix(B, M, K);
    copy_transposed(B, TB, M, K);
    aligned_double *C = create_matrix(N, K);

    struct timespec begin_time, end_time;

    memset(C, 0, N * K * sizeof(*C));

    clock_gettime(CLOCK_MONOTONIC, &begin_time);
    multiply_matrix(A, B, C, N, M, K);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    print_time("Single", begin_time, end_time);

    memset(C, 0, N * K * sizeof(*C));

    clock_gettime(CLOCK_MONOTONIC, &begin_time);
    multiply_matrix_omp1(A, B, C, N, M, K, T);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    print_time("OMP1", begin_time, end_time);

    memset(C, 0, N * K * sizeof(*C));

    clock_gettime(CLOCK_MONOTONIC, &begin_time);
    multiply_matrix_omp2(A, B, C, N, M, K, T);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    print_time("OMP2", begin_time, end_time);

    memset(C, 0, N * K * sizeof(*C));

    clock_gettime(CLOCK_MONOTONIC, &begin_time);
    multiply_matrix_tr(A, TB, C, N, M, K);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    print_time("Single TR", begin_time, end_time);

    memset(C, 0, N * K * sizeof(*C));

    clock_gettime(CLOCK_MONOTONIC, &begin_time);
    multiply_matrix_omp1_tr(A, TB, C, N, M, K, T);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    print_time("OMP1 TR", begin_time, end_time);

    memset(C, 0, N * K * sizeof(*C));

    clock_gettime(CLOCK_MONOTONIC, &begin_time);
    multiply_matrix_omp2_tr(A, TB, C, N, M, K, T);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    print_time("OMP2 TR", begin_time, end_time);

    free(A);
    free(B);
    free(C);
    free(TB);
}

void run(int SZ, int T) {
    int N = SZ, M = SZ, K = SZ;
    aligned_double *A = create_matrix(N, M);
    aligned_double *B = create_matrix(M, K);
    fill_matrix(A, N, M);
    fill_matrix(B, M, K);

    aligned_double *C = create_matrix(N, K);
    memset(C, 0, N * K * sizeof(*C));
    struct timespec begin_time, end_time;

    double result = 0;
    for (int i = 0; i < 3; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &begin_time);
        multiply_matrix_omp2(A, B, C, N, M, K, T);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        result += time_sec(begin_time, end_time) / 3.0;
    }

    printf("%d %d %f\n", SZ, T, result);

    free(A);
    free(B);
    free(C);    
}

int main(int argc, char **argv)
{
    if (argc == 1) {
        puts("No arguments passed");
        return 0;
    }

    if (strcmp("-t", argv[1]) == 0) {
        srand(1234);
        int passed = 0;
        int test_cnt = 5 * 3;
        passed += test(100);
        passed += test(500);
        passed += test(1000);
        printf("\n%d of %d passed\n", passed, test_cnt);
        if (passed == test_cnt) {
            puts("\x001b[32mSUCCESS\x001b[0m");
        } else {
            puts("\x001b[31mFAIL\x001b[0m");
        }
        return 0;
    }    

    if (strcmp("-c", argv[1]) == 0) {
        srand(1234);
        compare_versions(3000);
        return 0;
    }

    if (strcmp("-r", argv[1]) == 0) {
        if (argc < 5) {
            puts("Array size or number of threads not passed");
        } else {
            int sz, threads, seed;
            bool success = 
                        sscanf(argv[2], "%d", &sz) == 1 &&
                        sscanf(argv[3], "%d", &threads) == 1&&
                        sscanf(argv[4], "%d", &seed) == 1;
            if (!success) {
                puts("Wrong input data");
                return 0;
            }

            srand(seed);
            run(sz, threads);
        }
        return 0;
    }

    puts("Wrong input data");

    return 0;
}