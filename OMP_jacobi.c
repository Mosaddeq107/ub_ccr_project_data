#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_ITER 100000000
#define CONV_THRESHOLD 1e-6
#define N 1000  // Number of internal points in each dimension

int main() {
    double u[N+2][N+2], u_new[N+2][N+2];
    double h = 1.0 / (N + 1);
    int i, j, iter;
    double diff, max_diff;
    double start_time, end_time;

    // Initialize the grid and set boundary conditions
    for (i = 0; i < N+2; i++) {
        for (j = 0; j < N+2; j++) {
            if (i == 0 || j == 0) u[i][j] = 0.0;
            else if (i == N+1 || j == N+1) u[i][j] = 100.0;
            else u[i][j] = 50.0;  // Initial guess
        }
    }

    start_time = omp_get_wtime();  // Start the timer

    // Jacobi Iteration
    for (iter = 0; iter < MAX_ITER; iter++) {
        max_diff = 0.0;

        #pragma omp parallel for private(j) reduction(max:max_diff)
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                u_new[i][j] = (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] + h*h*-81*i*h*j*h) / 4.0;
                diff = fabs(u_new[i][j] - u[i][j]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        // Update u and check for convergence
        #pragma omp parallel for private(j)
        for (i = 1; i <= N; i++) {
            for (j = 1; j <= N; j++) {
                u[i][j] = u_new[i][j];
            }
        }


        if (max_diff < CONV_THRESHOLD) {
            break;
        }
    }

    end_time = omp_get_wtime();  // End the timer

    printf("Converged after %d iterations\n", iter);
    printf("Execution Time: %f seconds\n", end_time - start_time);  // Print the execution time

    // Optional: Print the solution
//    for (i = 0; i < N+2; i++) {
  //      for (j = 0; j < N+2; j++) {
    //        printf("%.2f ", u[i][j]);
      //  }
 //       printf("\n");
   // }

    return 0;
}

