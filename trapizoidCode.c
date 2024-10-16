#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Function to evaluate the curve (y = f(x))
float f(float x) {
    return x * x;  // Example: y = x^2
}

// Function to compute the area of a trapezoid
float trapezoid_area(float a, float b, float d) { 
    float area = 0;
    for (float x = a; x < b; x += d) {
        area += f(x) + f(x + d);
    }
    return area * d / 2.0f;
}

// Function to calculate the trapezoidal area serially for comparison
float serial_trapezoid(float a, float b, int n) {
    float d = (b - a) / n;
    float area = 0;
    for (float x = a; x < b; x += d) {
        area += f(x) + f(x + d);
    }
    return area * d / 2.0f;
}

int main(int argc, char** argv) {
    int rank, size;
    float a = 0.0f, b = 1.0f;  // Limits of integration
    int n;
    float start, end, local_area, total_area;
    double start_time, end_time, execution_time, serial_time, speedup, efficiency;

    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get number of processes

    if (rank == 0) {
        // Get the number of intervals from the user
        printf("Enter the number of intervals: ");
        scanf("%d", &n);

        // Serial execution (single processor equivalent)
        start_time = MPI_Wtime();
        float serial_area = serial_trapezoid(a, b, n);  // Perform serial computation
        end_time = MPI_Wtime();
        serial_time = end_time - start_time;  // Calculate serial execution time
        printf("Serial execution time: %f seconds\n", serial_time);
        printf("Serial area under the curve: %f\n", serial_area);
    }

    // Broadcast the number of intervals to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the interval size for each process
    float d = (b - a) / n;  // delta
    float region = (b - a) / size;

    // Calculate local bounds for each process
    start = a + rank * region;
    end = start + region;

    // Each process calculates the area of its subinterval
    local_area = trapezoid_area(start, end, d);

    // Reduce all local areas to the total area on the root process
    MPI_Reduce(&local_area, &total_area, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();  // End timing
    execution_time = end_time - start_time;

    if (rank == 0) {
        // Output the total time and total area under the curve
        printf("Execution time with %d processors: %f seconds\n", size, execution_time);
        printf("The total area under the curve is: %f\n", total_area);

        // Speedup and efficiency calculations
        speedup = serial_time / execution_time;  // Speed-up calculation
        efficiency = speedup / size;  // Efficiency calculation

        printf("Speedup with %d processors: %f\n", size, speedup);
        printf("Efficiency with %d processors: %f\n", size, efficiency);
    }

    MPI_Finalize();  // Finalize MPI
   return 0;
}
