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
    for (int j = 0; j < 8; j++) {
      fprintf(f, "%.8f\t", results[j]);
    }
    fprintf(f,"\n");
    free(results);

    //Printing the progress to stdout
    printf("\r%.0f%%", 100.0*(i+1)/n);
    fflush(stdout);
  }

  fclose(f);
  printf("\r");
}

/*Running t_len*sizes_len simulations with
different lattice sizes and temperatures*/
void run_multiple_sizes(double *T, int t_len, int *sizes, int sizes_len,
                        double J, int mc, int trans, char *file) {

  char str[100];

  for (int i = 1; i <= sizes_len; i++) {
    sprintf(str, "%s%d", file, i);
    run_n_simul(T, J, t_len, sizes[i-1], mc, trans, str);
    printf("Finished with %d/%d lattices\n", i, sizes_len);
  }

}

void run_plotting() {
  double J = -1;
  double *T = linspace(1,8,20);
  int t_len = 20;
  int sizes[] = {20,25,30,40};
  int sizes_len = 4;
  int mc = 5e2;
  int trans = 5e2;

 run_multiple_sizes(T, t_len, sizes, sizes_len, J, mc, trans, "output");

 free(T);
}
