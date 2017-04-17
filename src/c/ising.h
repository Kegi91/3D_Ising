int* initial_array(int n);
double E(int i, int j, int k, int *array, int n, double J);
double *boltzmann_factors(double T, double J);
int update_spin(int n, int i, int j, int k, int *array, double *b_factors, double J);
double *simulation(int n, int mc_steps, int trans_steps, double T, double J);
