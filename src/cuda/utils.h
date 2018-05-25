#include "pcg/pcg_basic.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

pcg32_random_t *seed(int n);
void allocate_rng_d(pcg32_random_t **rng, int n);
__host__ __device__ uint32_t rand_pcg(pcg32_random_t *rng);
__host__ __device__ uint32_t rand_pcg_bound(pcg32_random_t *rng, uint32_t lim);
__host__ __device__ float rand_f(pcg32_random_t *rng_all, int idx);
__host__ __device__ int rand_int(int lim, pcg32_random_t *rng_all, int idx);
int8_t *malloc_int_h(unsigned long n);
int *malloc_integer_h(unsigned long n);
void malloc_int_d(int8_t **array, unsigned long n);
void malloc_integer_d(int **array, unsigned long n);
float *malloc_float_h(unsigned long n);
void malloc_float_d(float **array, unsigned long n);
__device__ int index(int i, int j, int k, int n);
float *linspace(float min, float max, int n);
int *linspace_int(int sizes_min, int sizes_max, int step);
void gpuAssert(cudaError_t code, const char *file, int line);
