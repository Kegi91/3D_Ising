#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "utils.h"
#include "pcg/pcg_basic.h"

// Seeding n PRNGs on the host
pcg32_random_t *seed(int n) {
  long base_seed;
  base_seed = time(NULL);

  // Allocate the memory and check \rERRORs
  pcg32_random_t *rng = (pcg32_random_t *) malloc(n*sizeof(pcg32_random_t));
  if (rng == NULL) {
    fprintf(stderr, "%s\n", "\rERROR allocating memory:\tseed()");
    exit(1);
  }

  // Seed each PRNG with different seed
  for (int i = 0; i<n; i++) {
    pcg32_srandom_r(&rng[i], base_seed+i, (intptr_t) &rng[i]);
  }

  return rng;
}

void allocate_rng_d(pcg32_random_t **rng, int n) {
  if (cudaMalloc(rng, n*sizeof(pcg32_random_t)) != cudaSuccess) {
    gpuErrchk(cudaPeekAtLastError());
    exit(1);
  }
}

// Generate 32-bit unsigned random integer using the PCG algorithm
__host__ __device__ uint32_t rand_pcg(pcg32_random_t *rng) {
  uint64_t oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + rng->inc;
  uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint32_t rot = oldstate >> 59u;

  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__host__ __device__ uint32_t rand_pcg_bound(pcg32_random_t *rng, uint32_t lim) {
  uint32_t threshold = -lim % lim;

  for (;;) {
    uint32_t r = rand_pcg(rng);
    if (r >= threshold) return r % lim;
  }

}

// Returns a float in the range [0,1) from PRNG with index idx
__host__ __device__ float rand_f(pcg32_random_t *rng_all, int idx) {
  pcg32_random_t *rng = &rng_all[idx];
  return ldexpf(rand_pcg(rng), -32);
}

//Returns integer x in the range 0 <= x < lim from PRNG with index idx
__host__ __device__ int rand_int(int lim, pcg32_random_t *rng_all, int idx) {
  pcg32_random_t *rng = &rng_all[idx];
  return rand_pcg_bound(rng, lim);
}

// Allocate memory in the host
int8_t *malloc_int_h(unsigned long n) {
  int8_t *array = (int8_t *) malloc(n*n*n);
  if (array == NULL) {
    fprintf(
      stderr,
      "%s\n", "\rERROR allocating memory in the host:\tmalloc_int_h()"
    );
    exit(1);
  }

  return array;
}

int *malloc_integer_h(unsigned long n) {
  int *array = (int *) malloc(n*n*n*sizeof(int));
  if (array == NULL) {
    fprintf(
      stderr,
      "%s\n", "\rERROR allocating memory in the host:\tmalloc_int_h()"
    );
    exit(1);
  }

  return array;
}

// Allocate memory in the device
void malloc_int_d(int8_t **array, unsigned long n) {
  if (cudaMalloc(array, n*n*n) != cudaSuccess) {
    gpuErrchk(cudaPeekAtLastError());
    exit(1);
  }
}

void malloc_integer_d(int **array, unsigned long n) {
  if (cudaMalloc(array, n*n*n*sizeof(int)) != cudaSuccess) {
    gpuErrchk(cudaPeekAtLastError());
    exit(1);
  }
}

float *malloc_float_h(unsigned long n) {
  float *array = (float *) malloc(n*sizeof(float));
  if (array == NULL) {
    fprintf(
      stderr,
      "%s\n", "\rERROR allocating memory in the host:\tmalloc_float_h()"
    );
    exit(1);
  }

  return array;
}

void malloc_float_d(float **array, unsigned long n) {
  if (cudaMalloc(array, n*sizeof(float))) {
    gpuErrchk(cudaPeekAtLastError());
    exit(1);
  }
}

//converts 3D index to the actual 1D index
__device__ int index(int i, int j, int k, int n) {
  return i*n*n + j*n + k;
}

float *linspace(float min, float max, int n) {
    float *ret = malloc_float_h(n);

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

  int *array = (int *) malloc(((sizes_max-sizes_min)/step+1)*sizeof(int));
  if (array == NULL) {
    fprintf(stderr, "%s\n", "\rERROR allocating memory");
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

void gpuAssert(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(
        stderr,
        "\rGPUassert: %s %s %d\n", cudaGetErrorString(code), file, line
      );
   }
}
