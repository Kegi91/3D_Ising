#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "utils.h"
#include "ising.h"
#include "run.h"

//Running a n simulations with different temperatures T
void run_n_simul(
  float *T, float J, int n, int size,
  int mc, int trans, int calc, char *file
) {

  char path[100];
  sprintf(path, "../../output/%s.dat", file);
  FILE *f = fopen(path, "w");

  // Initializing in both host and device
  pcg32_random_t *rng_h, *rng_d;
  int rng_num = ceil(size*size*size / 2.0);
  rng_h = seed(rng_num);
  allocate_rng_d(&rng_d, rng_num);
  cudaMemcpy(
    rng_d, rng_h,
    rng_num * sizeof(pcg32_random_t),
    cudaMemcpyHostToDevice
  );

  float *results;

  for (int i = 0; i < n; i++) {
    results = run_simulation(size,mc,trans,calc,T[i],J,rng_h, rng_d);
    for (int j = 0; j < 9; j++) {
      fprintf(f, "%.8lf\t", results[j]);
    }
    fprintf(f,"\n");
    free(results);

    //Printing the progress to stdout
    printf("\r%.0lf%%", 100.0*(i+1)/n);
    fflush(stdout);
  }

  free(rng_h);
  cudaFree(rng_d);

  fclose(f);
  printf("\r");
}

/*Running t_len*sizes_len simulations with
different lattice sizes and temperatures*/
void run_multiple_sizes(
  float *T, int t_len, int *sizes, int sizes_len,
  float J, int mc, int trans, int calc, char *file
) {

  char str[100];

  for (int i = 1; i <= sizes_len; i++) {
    sprintf(str, "%s%d", file, i);
    run_n_simul(T, J, t_len, sizes[i-1], mc, trans, calc, str);
    printf("Finished with %d/%d lattices\n", i, sizes_len);
  }

}

void read_n_run(char *f_in) {
  FILE *fp = fopen(f_in,"r");
  if (fp == NULL) exit(1);

  char buffer[100];
  int J;
  float T_min, T_max;
  int T_len, sizes_min, sizes_max, step;
  int mc, trans, calc;
  char fname[100];

  // Reading the parameters
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp," %d ", &J) == 0) exit(1);
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp," %f %f %d ", &T_min, &T_max, &T_len) == 0) exit(1);
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp," %d %d %d ", &sizes_min, &sizes_max, &step) == 0) exit(1);
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp," %d %d %d ", &mc, &trans, &calc) == 0) exit(1);
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp,"%s", fname) == 0) exit(1);
  fclose(fp);

  float *T = linspace(T_min, T_max, T_len);
  int *sizes = linspace_int(sizes_min, sizes_max, step);
  int sizes_len = (sizes_max-sizes_min)/step+1;

  run_multiple_sizes(T, T_len, sizes, sizes_len, J, mc, trans, calc, fname);

  free(sizes);
  free(T);

  gpuErrchk(cudaPeekAtLastError());
}

 void benchmark(char f_in[]) {

   // Reading all the parameters
   FILE *fp = fopen(f_in,"r");
   if (fp == NULL) exit(1);

   char buffer[100];
   int J;
   float T_min, T_max;
   int T_len, sizes_min, sizes_max, step;
   int mc, trans, calc;
   char fname[100];

   // Reading the parameters
   if (fgets(buffer, 100, fp) == NULL) exit(1);
   if (fscanf(fp," %d ", &J) == 0) exit(1);
   if (fgets(buffer, 100, fp) == NULL) exit(1);
   if (fscanf(fp," %f %f %d ", &T_min, &T_max, &T_len) == 0) exit(1);
   if (fgets(buffer, 100, fp) == NULL) exit(1);
   if (fscanf(fp," %d %d %d ", &sizes_min, &sizes_max, &step) == 0) exit(1);
   if (fgets(buffer, 100, fp) == NULL) exit(1);
   if (fscanf(fp," %d %d %d ", &mc, &trans, &calc) == 0) exit(1);
   if (fgets(buffer, 100, fp) == NULL) exit(1);
   if (fscanf(fp,"%s", fname) == 0) exit(1);
   fclose(fp);

   float *T = linspace(T_min, T_max, T_len);
   int *sizes = linspace_int(sizes_min, sizes_max, step);
   int sizes_len = (sizes_max-sizes_min)/step+1;

   struct timespec start, stop;
   double accum;

   // Sizes
   for (int i = 0; i<sizes_len; i++) {
     // Initializing in both host and device
     pcg32_random_t *rng_h, *rng_d;
     int rng_num = ceil(sizes[i]*sizes[i]*sizes[i] / 2.0);
     rng_h = seed(rng_num);
     allocate_rng_d(&rng_d, rng_num);
     cudaMemcpy(
       rng_d, rng_h,
       rng_num * sizeof(pcg32_random_t),
       cudaMemcpyHostToDevice
     );

     clock_gettime(CLOCK_MONOTONIC, &start);

     // Temps
     for (int j = 0; j < T_len; j++) {

      //  Average over 10 repeats
       for (int repeat = 0; repeat < 10; repeat++) {
         run_simulation(sizes[i], mc, trans, calc, T[j], J, rng_h, rng_d);
       }

     }

     clock_gettime(CLOCK_MONOTONIC, &stop);
     accum = (stop.tv_sec-start.tv_sec) + (stop.tv_nsec-start.tv_nsec) / 1e9;
     accum /= 10;

     printf("%d\t%lf\n", sizes[i], accum);

     free(rng_h);
     cudaFree(rng_d);
   }

   free(sizes);
   free(T);
 }
