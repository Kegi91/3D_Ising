#include <stdlib.h>
#include <stdio.h>

#include "pcg/pcg_basic.h"
#include "utils.h"
#include "ising.h"
#include "run.h"

//Running a n simulations with different temperatures T
void run_n_simul(
  double *T, double J, int n, int size,
  int mc, int calc, int trans, char *file
) {

  char path[100];
  sprintf(path, "../../output/%s.dat", file);
  FILE *f = fopen(path, "w");

  pcg32_random_t rng;
  seed(&rng);

  double *results;

  for (int i = 0; i < n; i++) {
    results = simulation(size,mc,trans,calc,T[i],J,rng);
    for (int j = 0; j < 9; j++) {
      fprintf(f, "%.8lf\t", results[j]);
    }
    fprintf(f,"\n");
    free(results);

    //Printing the progress to stdout
    printf("\r%.0lf%%", 100.0*(i+1)/n);
    fflush(stdout);
  }

  fclose(f);
  printf("\r");
}

/*Running t_len*sizes_len simulations with
different lattice sizes and temperatures*/
void run_multiple_sizes(
  double *T, int t_len, int *sizes, int sizes_len,
  double J, int mc, int calc, int trans, char *file
) {

  char str[100];

  for (int i = 1; i <= sizes_len; i++) {
    sprintf(str, "%s%d", file, i);
    run_n_simul(T, J, t_len, sizes[i-1], mc, trans, calc, str);
    printf("Finished with %d/%d lattices\n", i, sizes_len);
  }

}

void read_n_run() {
  FILE *fp = fopen("../../input/input.dat","r");
  if (fp == NULL) exit(1);

  char buffer[100];
  int J;
  double T_min, T_max;
  int T_len, sizes_min, sizes_max, step;
  int mc,trans,calc;
  char fname[100];

  // Reading the parameters
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp," %d ", &J) == 0) exit(1);
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp," %lf %lf %d ", &T_min, &T_max, &T_len) == 0) exit(1);
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp," %d %d %d ", &sizes_min, &sizes_max, &step) == 0) exit(1);
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp," %d %d %d ", &mc, &trans, &calc) == 0) exit(1);
  if (fgets(buffer, 100, fp) == NULL) exit(1);
  if (fscanf(fp,"%s", fname) == 0) exit(1);
  fclose(fp);

  double *T = linspace(T_min, T_max, T_len);
  int *sizes = linspace_int(sizes_min, sizes_max, step);
  int sizes_len = (sizes_max-sizes_min)/step+1;

  run_multiple_sizes(T, T_len, sizes, sizes_len, J, mc, trans, calc, fname);

  free(sizes);
  free(T);
}
