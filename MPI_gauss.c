#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 100000
#define CONV_THRESHOLD 1e-6
#define N 200  // Ensure N is divisible by the number of MPI processes

void initializeGrid(double *u, int rows, int my_rank, int size) {
    for (int i = 0; i < rows + 2; ++i) {
        for (int j = 0; j < N + 2; ++j) {
            if (i == 0 || i == rows + 1 || j == 0 || j == N + 1) {
                // Set boundary conditions
                u[i * (N + 2) + j] = 100.0 * (my_rank == 0 && i == 0) + 100.0 * (my_rank == size - 1 && i == rows + 1);
            } else {
                u[i * (N + 2) + j] = 50.0;  // Initial guess
            }
        }
    }
}

void updateRedBlack(double *u, int rows, double h, double *max_diff, int color) {
    double diff, temp;
    *max_diff = 0.0;

    for (int i = 1; i <= rows; ++i) {
        for (int j = 1; j <= N; ++j) {
            if ((i + j + color) % 2 == 0) {
                int idx = i * (N + 2) + j;
                temp = u[idx];
                u[idx] = (u[idx - 1] + u[idx + 1] + u[idx - (N + 2)] + u[idx + (N + 2)] + h * h * -81 * (i + color) * h * j * h) / 4.0;
                diff = fabs(u[idx] - temp);
                if (diff > *max_diff) *max_diff = diff;
            }
        }
    }
}

int main(int argc, char **argv) {
    int rank, size, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    double h = 1.0 / (N + 1);
    double *u = (double *)malloc((rows_per_proc + 2) * (N + 2) * sizeof(double));
    double max_diff, global_max_diff;

    initializeGrid(u, rows_per_proc, rank, size);

    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Communicate ghost layers
        MPI_Request request[4];
        if (rank != 0) {
            MPI_Isend(&u[(N + 2)], N + 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(&u[0], N + 2, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &request[1]);
        }
        if (rank != size - 1) {
            MPI_Isend(&u[rows_per_proc * (N + 2)], N + 2, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &request[2]);
            MPI_Irecv(&u[(rows_per_proc + 1) * (N + 2)], N + 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &request[3]);
        }

        // Update Red Points
        updateRedBlack(u, rows_per_proc, h, &max_diff, 0);

        // Wait for communication to finish
        if (rank != 0) {
            MPI_Wait(&request[0], MPI_STATUS_IGNORE);
            MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        }
        if (rank != size - 1) {
            MPI_Wait(&request[2], MPI_STATUS_IGNORE);
            MPI_Wait(&request[3], MPI_STATUS_IGNORE);
        }

        // Update Black Points
        updateRedBlack(u, rows_per_proc, h, &max_diff, 1);

        // Check for convergence
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (global_max_diff < CONV_THRESHOLD) {
            if (rank == 0) {
                end_time = MPI_Wtime();
                printf("Converged after %d iterations\n", iter + 1);
                printf("Execution Time: %f seconds\n", end_time - start_time);
            }
            break;
        }
    }

    // Clean up
    free(u);
    MPI_Finalize();
    return 0;
}
