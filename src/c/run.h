void run_n_simul(
  double *T, double J, int n, int size,
  int mc, int trans, int calc, char* file
);
void run_multiple_sizes(
  double *T, int t_len, int *sizes, int sizes_len,
  double J, int mc, int trans, int calc, char *file
);
void read_n_run();
void benchmark(char f_in[]);
