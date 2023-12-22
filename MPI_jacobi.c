#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 4  // Number of points including boundaries
#define MAX_ITER 100000
#define TOL 1e-6

double f(double x, double y) {
    return -81 * x * y;
}

int main(int argc, char *argv[]) {
    int rank, size, i, j, iter;
    double u[N][N], new_u[N][N], diff, max_diff, x, y, h;
    double start_time, end_time;  // Variables for measuring execution time

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize parameters
    h = 1.0 / (N - 1);

    // Initialize u
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == 0 || j == 0) u[i][j] = 0;
            else if (i == N - 1 || j == N - 1) u[i][j] = 100;
            else u[i][j] = 0;
        }
    }

    // Jacobi Iteration
    double local_max_diff;
    start_time = MPI_Wtime();  // Record start time
    for (iter = 0; iter < MAX_ITER; iter++) {
        local_max_diff = 0;

        // Update new_u based on u
        for (i = 1; i < N - 1; i++) {
            for (j = 1; j < N - 1; j++) {
                x = i * h;
                y = j * h;
                new_u[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - h*h*f(x, y));

                diff = fabs(new_u[i][j] - u[i][j]);
                if (diff > local_max_diff) local_max_diff = diff;
            }
        }

        // Check for convergence
        MPI_Allreduce(&local_max_diff, &max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (max_diff < TOL) break;

        // Update u for the next iteration
        for (i = 1; i < N - 1; i++)
            for (j = 1; j < N - 1; j++)
                u[i][j] = new_u[i][j];
    }
    end_time = MPI_Wtime();  // Record end time

    if (rank == 0) {
        printf("Jacobi iterations: %d\n", iter);
        printf("Execution time: %f seconds\n", end_time - start_time);  // Print execution time
        // Print final u values or write them to a file
    }

    MPI_Finalize();
    return 0;
}

