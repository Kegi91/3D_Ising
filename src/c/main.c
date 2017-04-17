#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ising.h"
#include "utils.h"
#include "pcg/pcg_basic.h"

int main() {
  pcg32_random_t rng;
  seed(&rng);
  int n = 3*3*3;

  double J = -1;
  int spins[n];
  for (int i = 0; i < n; i++) {
    spins[i] = -1;
  }

  //testing single flip. The spin should be flipped with ~0.5134 probability
  int flipped = 0;
  double *b_factors = boltzmann_factors(18,J);

  for (int i = 0; i < 2e6; i++) {
    if (update_spin(3,2,2,2,spins,b_factors,J,&rng)) {
      flipped++;
      spins[index(2,2,2,3)] = -1;
    }
  }

  free(b_factors);
  printf("%f\n", (double)flipped/2e6);

  return 0;
}
