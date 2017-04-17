#include <time.h>
#include <stdlib.h>
#include <stdio.h>

void seed() {
  //The initial seed. Should be called only once per run
  srand(time(NULL));
}

double rand_d() {
  return (double)rand()/(double)RAND_MAX;
}

/*allocating the memory for the array
  1D array is used to make sure the array is stored
  in a contiguous block of memory*/
int *malloc_int(int n) {
  int *array = malloc(n*n*n*sizeof(int));
  if (array == NULL) {
    fprintf(stderr, "%s\n", "ERROR allocating memory");
  }
  return array;
}

double *malloc_double(int n) {
  double *array = malloc(n*n*n*sizeof(double));
  if (array == NULL) {
    fprintf(stderr, "%s\n", "ERROR allocating memory");
  }
  return array;
}

//converts 3D index to the actual 1D index
int index(int i, int j, int k, int n) {
  return i*n*n + j*n + k;
}
