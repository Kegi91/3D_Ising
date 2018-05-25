#include "pcg/pcg_basic.h"

int8_t *initial_array(unsigned long n);
int8_t *initial_array_random(unsigned long n, pcg32_random_t *rng);
__device__ float E(int i, int j, int k, int8_t *array, int n, float J);
__global__ void E_total(int8_t *array, int n, float J, float *E_result_d);
__global__ void M_total(int8_t *array, int n, int *M_result_d);
float *boltzmann_factors(float T, float J);
__global__ void update_spins(
  int n, int8_t *array, float *b_factors,
  float J, int which, pcg32_random_t *rng
);
float *run_simulation(
  int n, int mc_steps, int trans_steps, int calc, float T, float J,
  pcg32_random_t *rng_h, pcg32_random_t *rng_d
);
