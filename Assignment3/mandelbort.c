#include <stdio.h> 
#include <time.h>
#include <omp.h>  // Include OpenMP header

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
  double real;
  double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;

    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;

        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n"); // Magic Number
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT); // Width and Height
    fprintf(pgmimg, "255\n"); // Max gray value

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main() {
    int image[HEIGHT][WIDTH];
    double AVG = 0;
    int N = 10;
    double total_time[N];

    for (int k = 0; k < N; k++) {
        clock_t start_time = clock();

        // Parallelize the computation loop
        #pragma omp parallel for schedule(dynamic, 1) shared(image) private(c)
        for (int i = 0; i < HEIGHT; i++) {
            struct complex c;
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[i][j] = cal_pixel(c);
            }
        }

        clock_t end_time = clock();
        total_time[k] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        printf("Execution time of trial [%d]: %f seconds\n", k + 1, total_time[k]);
        AVG += total_time[k];
    }

    save_pgm("mandelbrot.pgm", image);
    printf("The average execution time of 10 trials is: %f ms\n", AVG / N * 1000);

    return 0;
}
