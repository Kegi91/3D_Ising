#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ising.h"
#include "utils.h"

int main() {
  seed();
  int n = 3*3*3;

  double J = -1;
  int spins[n];
  for (int i = 0; i < n; i++) {
    spins[i] = -1;
  }

  //testing single flip. The spin should be flipped with ~0.5134 probability
  int flipped = 0;
  double *b_factors = boltzmann_factors(18,J);

  for (int i = 0; i < 100000; i++) {
    if (update_spin(3,2,2,2,spins,b_factors,J)) {
      flipped++;
      spins[index(2,2,2,3)] = -1;
    }
  }

  printf("%f\n", (double)flipped/100000);

  return 0;
}
