#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "run.h"
#include "utils.h"

#define functions_len 1
char functions[functions_len][100] = {
  "run_from_input: input_file"
};

void err_msg();
void run_from_input(char f_in[]);

int main(int argc, char **argv) {
  if (argc < 3) {
    err_msg();
    exit(0);
  }

  if (strcmp(argv[argc-2], "run_from_input") == 0) run_from_input(argv[argc-1]);

  return 0;
}

void err_msg() {
  fprintf(
    stderr,
    "%s\n\n%s\n\n%s\n\n",
    "Give function and parameters as cmd line parameters e.g.",
    "./main run_from_input ../../input/input.dat",
    "The supported functions and parameters are [function: ...]:"
  );

  for (int i = 0; i < functions_len; i++) {
    fprintf(stderr, "%s\n", functions[i]);
  }
}

void run_from_input(char *f_in) {
  read_n_run(f_in);
}
