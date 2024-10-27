#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

void pipelined_parallel_sieve(int rank, int size, int n, double *parallel_time) {
    int candidate = 2;  // Starting prime candidate
    int current_prime = -1;
    bool found_prime = false;  // Flag for whether the process has found a prime
    MPI_Status status;

    double start_time = MPI_Wtime();  // Start time for parallel execution

    int upper_bound = n * n;  // Increase as necessary

    if (rank == 0) {
        current_prime = candidate;
        printf("Process %d found prime: %d\n", rank, current_prime);
        candidate++;

        for (; candidate <= upper_bound; candidate++) {
            if (candidate % current_prime != 0) {  
                MPI_Send(&candidate, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            }
        }

        candidate = -1;
        MPI_Send(&candidate, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    } else {
        while (true) {
            MPI_Recv(&candidate, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);

            if (candidate == -1) {  // Termination signal
                if (rank < size - 1) {
                    MPI_Send(&candidate, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                }
                break;
            }

            if (!found_prime) {
                current_prime = candidate;
                found_prime = true;
                printf("Process %d found prime: %d\n", rank, current_prime);
            } else {
                if (candidate % current_prime != 0) {
                    if (rank < size - 1) {
                        MPI_Send(&candidate, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }
    }

    *parallel_time = MPI_Wtime() - start_time;  
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 100;  
    double parallel_time, max_parallel_time;
    double sequential_time, speedup, efficiency;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        sequential_time = MPI_Wtime();
    }

    pipelined_parallel_sieve(rank, size, n, &parallel_time);

    if (rank == 0) {
        sequential_time = MPI_Wtime() - sequential_time;
    }

    MPI_Reduce(&parallel_time, &max_parallel_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        speedup = sequential_time / max_parallel_time;
        efficiency = speedup / size;

        printf("\nSequential Time: %f seconds\n", sequential_time);
        printf("Parallel Time: %f seconds\n", max_parallel_time);
        printf("Speedup: %f\n", speedup);
        printf("Efficiency: %f\n", efficiency);
    }

    MPI_Finalize();
    return 0;
}
