#include <stdlib.h>
#include <stdio.h>

#include "pcg/pcg_basic.h"
#include "utils.h"
#include "ising.h"
#include "run.h"

//Running a n simulations with different temperatures T
void run_n_simul(double *T, double J, int n, int size, int mc, int trans, char *file) {
  char path[100];
  sprintf(path, "../../output/%s.dat", file);
  FILE *f = fopen(path, "w");

  pcg32_random_t rng;
  seed(&rng);

  double *results;

  for (int i = 0; i < n; i++) {
    results = simulation(size,mc,trans,T[i],J,rng);
    for (int j = 0; j < 5; j++) {
      fprintf(f, "%.8f\t", results[j]);
    }
    fprintf(f,"\n");
    free(results);
  }

  fclose(f);
}

/*Running t_len*sizes_len simulations with
different lattice sizes and temperatures*/
void run_multiple_sizes(double *T, int t_len, int *sizes, int sizes_len,
                        double J, int mc, int trans, char *file) {

  char str[100];

  for (int i = 1; i <= sizes_len; i++) {
    sprintf(str, "%s%d", file, i);
    run_n_simul(T, J, t_len, sizes[i-1], mc, trans, str);
  }

}
