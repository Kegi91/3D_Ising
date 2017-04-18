#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ising.h"
#include "utils.h"
#include "pcg/pcg_basic.h"

//Testing the function simulation()
int main() {
  pcg32_random_t rng;
  seed(&rng);
  int n = 10;
  double J = -1;

  double T[] = {1,2,3,4,5,6,7,8,9,10};
  double *results;

  for (int i = 0; i < 10; i++) {
    results = simulation(n,5e4,1e4,T[i],J,rng);
    for (int j = 0; j < 5; j++) {
      printf("%f\t", results[j]);
    }
    printf("\n");
  }

  return 0;
}
