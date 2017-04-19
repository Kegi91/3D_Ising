#include "pcg/pcg_basic.h"

int *initial_array(int n);

int* initial_array_random(int n, pcg32_random_t *rng);

double E(int i, int j, int k, int *array, int n, double J);

double E_total(int *array, int n, double J);

int M_total(int *array, int n);

double *boltzmann_factors(double T, double J);

int update_spin(int n, int i, int j, int k, int *array,
                double *b_factors, double J, pcg32_random_t *rng);

double *simulation(int n, int mc_steps, int trans_steps, double T,
                   double J, pcg32_random_t rng);
