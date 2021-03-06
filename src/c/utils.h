#include "pcg/pcg_basic.h"

void seed(pcg32_random_t *rng);
double rand_d(pcg32_random_t *rng);
int rand_int(int lim, pcg32_random_t *rng);
int *malloc_int(int n);
double *malloc_double(int n);
int index(int i, int j, int k, int n);
double *linspace(double min, double max, int n);
int *linspace_int(int sizes_min, int sizes_max, int step);
