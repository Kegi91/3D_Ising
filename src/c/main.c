#include "run.h"

//Testing the function simulation()
int main() {
  int size = 10;
  double J = -1;
  double T[] = {1,2,3,4,5,6,7,8,9,10};
  int t_len = 10;
  int sizes[] = {2,4,8,16};
  int sizes_len = 4;
  int mc = 1e3;
  int trans = 1e3;

  run_n_simul(T, J, t_len, size, mc, trans, "output");
//  run_multiple_sizes(T, t_len, sizes, sizes_len, J, mc, trans, "output");

  return 0;
}
