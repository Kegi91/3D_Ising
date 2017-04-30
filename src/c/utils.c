#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "pcg/pcg_basic.h"

//Seeding the PRNG
void seed(pcg32_random_t *rng) {
  pcg32_srandom_r(rng, time(NULL), (intptr_t)rng);
}

//Returns a double in the range [0,1)
double rand_d(pcg32_random_t *rng) {
  return ldexp(pcg32_random_r(rng), -32);
}

//Returns integer x in the range 0 <= x < lim
int rand_int(int lim, pcg32_random_t *rng) {
  return pcg32_boundedrand_r(rng, lim);
}

/*allocating the memory for the array
  1D array is used to make sure the array is stored
  in a contiguous block of memory*/
int *malloc_int(int n) {
  int *array = malloc(n*n*n*sizeof(int));
  if (array == NULL) {
    fprintf(stderr, "%s\n", "ERROR allocating memory");
    exit(1);
  }
  return array;
}

double *malloc_double(int n) {
  double *array = malloc(n*sizeof(double));
  if (array == NULL) {
    fprintf(stderr, "%s\n", "ERROR allocating memory");
    exit(1);
  }
  return array;
}

//converts 3D index to the actual 1D index
int index(int i, int j, int k, int n) {
  return i*n*n + j*n + k;
}

double *linspace(double min, double max, int n) {
    double *ret = malloc_double(n);

    for (int i = 0; i < n; i++){
        ret[i] = min + i*(max-min)/(n-1);
    }

    return ret;
}

int *linspace_int(int sizes_min, int sizes_max, int step) {
  if (sizes_min > sizes_max) {
    fprintf(stderr, "%s\n", "Invalid number of lattices");
    exit(1);
  }

  int *array = malloc(((sizes_max-sizes_min)/step+1)*sizeof(int));
  if (array == NULL) {
    fprintf(stderr, "%s\n", "ERROR allocating memory");
    exit(1);
  }

  int i = 0;
  while (sizes_min <= sizes_max) {
    array[i] = sizes_min;
    i++;
    sizes_min += step;
  }

  return array;
}
