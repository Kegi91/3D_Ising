#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ising.h"
#include "utils.h"

int main() {
  seed();
  int *array = initial_array(200);

  /*testing the function E()
    sum over random spin configurations energies divided
    by number of spins should be ~0
  */

  int sum = 0;
  for (int i = 0; i<200; i++) {
    for (int j = 0; j<200; j++) {
      for (int k = 0; k<200; k++) {
        sum += E(i,j,k,array,200,-1);
      }
    }
  }

  free(array);
  printf("%f\n", sum/pow(200,3));
  return 0;
}
