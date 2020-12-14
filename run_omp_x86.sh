#!/bin/bash

echo "" > result_omp_x86_1

for n in {100..1000..500}
do
    for t in {0..3..1}
    do
        ./omp -r $n $((2 ** $t)) 1 >> result_omp_x86_1
    done
done


# echo "" > result_omp_x86_2d

# for t in {1..12}
# do
#     ./omp -r 4000 $t 1 >> result_omp_x86_2d
# done