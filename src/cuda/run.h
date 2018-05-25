void run_n_simul(
  float *T, float J, int n, int size,
  int mc, int trans, int calc, char* file
);
void run_multiple_sizes(
  float *T, int t_len, int *sizes, int sizes_len,
  float J, int mc, int trans, int calc, char *file
);
void read_n_run(char *f_in);
void benchmark(char f_in[]);
