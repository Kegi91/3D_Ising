#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "run.h"
#include "utils.h"

#define functions_len 1
char functions[functions_len][100] = {
  "read: input_file"
};

void err_msg();

int main(int argc, char **argv) {
  if (argc < 3) {
    err_msg();
    exit(0);
  }

  if (strcmp(argv[argc-2], "read") == 0) read_n_run(argv[argc-1]);
  else if (strcmp(argv[argc-2], "benchmark") == 0) benchmark(argv[argc-1]);
  else err_msg();

  return 0;
}

void err_msg() {
  fprintf(
    stderr,
    "%s\n\n%s\n\n%s\n\n",
    "Give function and parameters as cmd line parameters e.g.",
    "./main read ../../input/input.dat",
    "The supported functions and parameters are [function: ...]:"
  );

  for (int i = 0; i < functions_len; i++) {
    fprintf(stderr, "%s\n", functions[i]);
  }
}
