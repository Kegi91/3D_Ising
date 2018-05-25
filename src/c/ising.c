#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "pcg/pcg_basic.h"
#include "ising.h"

//initializing array with all spins 1
int *initial_array(int n) {
  int *array = malloc_int(n);

  for (int i = 0; i<n*n*n; i++) {
    array[i] = 1;
  }

  return array;
}

//initializing the array with random spin configuration
int *initial_array_random(int n, pcg32_random_t *rng) {
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

  return -1*J*array[index(i,j,k,n)]*(left+right+up+down+front+back);
}

//Total energy of the lattice
double E_total(int *array, int n, double J) {
  double E_tot = 0;

  //summing over all spins
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        E_tot += E(i,j,k,array,n,J);
      }
    }
  }

  return E_tot;
}

int M_total(int *array, int n) {
  int size = n*n*n;
  int M_tot = 0;

  for (int i = 0; i < size; i++) {
      M_tot += array[i];
  }

  return M_tot;
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

  double b_factor;
  if (idx > 0) { //to not call b_factors[-1]
    b_factor = b_factors[idx-1];
  } else {
    b_factor = 0;
  }

  if (En > 0 || rand_d(rng) < b_factor) {
    array[index(i,j,k,n)] = array[index(i,j,k,n)] * -1;
    return 1;
  }

  return 0;
}

//A single simulation
double *simulation(
  int n, int mc_steps, int trans_steps, int calc,
  double T, double J, pcg32_random_t rng
) {

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

  int *spins = initial_array(n);
  double *b_factors = boltzmann_factors(T,J);
  int i,j,k;

  int samples = 0;

  //Transient mc steps
  for (int mc = 0; mc < trans_steps; mc++) {
    for (int m = 0; m < n*n*n; m++) {
      i = rand_int(n, &rng);
      j = rand_int(n, &rng);
      k = rand_int(n, &rng);
      update_spin(n,i,j,k,spins,b_factors,J,&rng);
    }
  }

  //Calculating the physical quantities before starting the main loop
  double E_curr = E_total(spins, n, J);
  double M_curr = M_total(spins, n);
  double E_tot = 0, Mabs_tot = 0, E2_tot = 0, M2_tot = 0, M4_tot = 0;

  //Main loop where the averages are calculated
  for (int mc = 0; mc < mc_steps; mc++) {
    for (int m = 0; m < n*n*n; m++) {
      i = rand_int(n, &rng);
      j = rand_int(n, &rng);
      k = rand_int(n, &rng);

      //Updating a single spin and the physical quantities
      if (update_spin(n,i,j,k,spins,b_factors,J,&rng)) {
        E_curr += 2*E(i, j, k, spins, n, J);
        M_curr += 2*spins[index(i,j,k,n)];
      }
    }

    // Updating physical quantities every calc mc steps
    if (mc % calc == 0) {
      E_tot += E_curr/3;
      E2_tot += pow(E_curr/3,2);
      Mabs_tot += fabs(M_curr);
      M2_tot += pow(M_curr, 2);
      M4_tot += pow(M_curr, 4);

      samples++;
    }
  }

  double norm = 1.0/(n*n*n*samples);
  double *ret = malloc_double(9);
  ret[0] = E_tot*norm; //mean energy
  ret[1] = E2_tot*norm; //mean energy^2
  ret[2] = Mabs_tot*norm; //mean |magnetization|
  ret[3] = M2_tot*norm; //mean magnetization^2
  ret[4] = M4_tot*norm; //mean magnetization^4
  ret[5] = (ret[1]-(ret[0]*ret[0]*n*n*n))/T; //heat capacity
  ret[6] = (ret[3]-(ret[2]*ret[2]*n*n*n))/T; //magnetic susceptibility
  ret[7] = 1-(ret[4]/(3*ret[3]*ret[3]*n*n*n)); //binder cumulant
  ret[8] = T; //temperature

  free(spins);
  free(b_factors);
  return ret;
}
