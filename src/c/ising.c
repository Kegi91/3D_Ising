#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "pcg/pcg_basic.h"
#include "ising.h"

//initializing the array with random spin configuration
int *initial_array(int n, pcg32_random_t *rng) {
  int *array = malloc_int(n);

  for (int i = 0; i<n*n*n; i++) {
    if (rand_d(rng) > 0.5) {
      array[i] = 1;
    } else {
      array[i] = -1;
    }
  }

  return array;
}

//Energy of a single spin
double E(int i, int j, int k, int *array, int n, double J) {
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

/*Returns an array of all the Boltzmann factors
  corresponding to different energies E<0.
  The factors are calculated beforehand to avoid
  calling exponential function on every iteration
  of the main loop*/
double *boltzmann_factors(double T, double J) {
  double *bf = malloc_double(6);

  //Loop over different energies E=J*i < 0
  if (J > 0) {
    for (int i = -6; i<0; i++) {
      bf[-1*(i+1)] = exp(2*J*i/T);
    }
  } else {
    for (int i = 6; i>0; i--) {
      bf[i-1] = exp(2*J*i/T);
    }
  }

  return bf;
}

/*Flipping a single spin according to the Metropolis algorithm
  Returns 1 of spin is flipped and 0 if not*/
int update_spin(int n, int i, int j, int k,
                int *array, double *b_factors, double J, pcg32_random_t *rng) {

  double En = E(i,j,k,array,n,J);
  int idx = fabs(En/J)+0.1; //+0.1 to negotiate floating point inaccuracy
  double b_factor = b_factors[idx-1];

  if (En > 0 || rand_d(rng) < b_factor) {
    array[index(i,j,k,n)] = array[index(i,j,k,n)] * -1;
    return 1;
  }

  return 0;
}

//A single simulation
double *simulation(int n, int mc_steps, int trans_steps, double T, double J) {
  /*This function initializes the spin array and runs the simulation
    using Metropolis algorithm to find the minimum of the free energy.

    First transient mc steps are run after which the system is assumed to be
    at equilibrium with the heat bath. After that the mc steps are run and
    the means of the needed physical quantities are calculated and returned.

    Notice: Seed needs to be given for the PRNG before calling this function.
            Seed should be given only once and can be done by calling the
            function seed() of utils.h.

    Parameters:

    n =           Cubic root of the number of the spin array elements.
                  i.e. lenght of an edge

    mc_steps =    Number of single spin flips = mc_steps * n^3

    trans_steps = Number of mc_steps before starting to calculate the means.
                  The system needs to be near the equilibrium after trans_steps
                  are run.

    T =           The temperature of the system

    J =           The coupling constant of the spin interaction
  */

  pcg32_random_t rng;
  seed(&rng);
  /*If multiple simulations are run the PRNG is seeded each time
    Since the seed uses both system time and the address of the pointer &rng
    it is highly unlikely the same seed is given*/

  int *spins = initial_array(n, &rng);
  double *b_factors = boltzmann_factors(T,J);

  //Transient steps
  for (int i = 0; i < trans_steps; i++) {
    for (int j = 0; j < n*n*n; j++) {
      //update_spin();
    }
  }

  free(spins);
  return NULL;
}
