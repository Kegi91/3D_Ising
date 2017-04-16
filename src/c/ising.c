#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "ising.h"

//initializing the array with random spin configuration
int *initial_array(int n) {
  int *array = malloc_int(n);

  for (int i = 0; i<n*n*n; i++) {
    if (rand_d() > 0.5) {
      array[i] = 1;
    } else {
      array[i] = -1;
    }
  }

  return array;
}

//Energy of a single spin
int E(int i, int j, int k, int *array, int n, double J) {
  int up,down,left,right,front,back; //neighbour spins

  //Checking the periodic boundary conditions
  if (i == n-1) {
    down = array[index(0,j,k,n)];
  } else {
    down = array[index(i+1,j,k,n)];
  }

  if (j == n-1) {
    right = array[index(i,0,k,n)];
  } else {
    right = array[index(i,j+1,k,n)];
  }

  if (k == n-1) {
    back = array[index(i,j,0,n)];
  } else {
    back = array[index(i,j,k+1,n)];
  }

  if (i == 0) {
    up = array[index(n-1,j,k,n)];
  } else {
    up = array[index(i-1,j,k,n)];
  }

  if (j == 0) {
    left = array[index(i,n-1,k,n)];
  } else {
    left = array[index(i,j-1,k,n)];
  }

  if (k == 0) {
    front = array[index(i,j,n-1,n)];
  } else {
    front = array[index(i,j,k-1,n)];
  }

  return J*array[index(i,j,k,n)]*(left+right+up+down+front+back);
}
