#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "utils.h"
#include "pcg/pcg_basic.h"
#include "ising.h"

//initializing array with all spins 1
int8_t *initial_array(unsigned long n) {
  int8_t *array = malloc_int_h(n);

  for (unsigned long i = 0; i<n*n*n; i++) {
    array[i] = 1;
  }

  return array;
}

//initializing the array with random spin configuration
int8_t *initial_array_random(unsigned long n, pcg32_random_t *rng) {
  int8_t *array = malloc_int_h(n);

  for (unsigned long i = 0; i<n*n*n; i++) {
    if (rand_f(rng, 0) > 0.5) {
      array[i] = 1;
    } else {
      array[i] = -1;
    }
  }

  return array;
}

//Energy of a single spin
__device__ float E(int i, int j, int k, int8_t *array, int n, float J) {
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
__global__ void E_total(int8_t *array, int n, float J, float *E_result_d) {
  int x,y,z;

  x = blockIdx.x;
  y = blockIdx.y;
  z = threadIdx.x;

  float e = E(x,y,z, array, n, J);

  __shared__ float blockSum;
  if (z == 0) blockSum = 0;
  __syncthreads();

  atomicAdd(&blockSum, e);
  __syncthreads();

  if (z == 0) atomicAdd(E_result_d, blockSum);
}

__global__ void M_total(int8_t *array, int n, float *M_result_d) {
  int x,y,z, idx;

  x = blockIdx.x;
  y = blockIdx.y;
  z = threadIdx.x;
  idx = index(x,y,z, n);

  float m = array[idx];

  __shared__ float blockSum;
  if (z == 0) blockSum = 0;
  __syncthreads();

  atomicAdd(&blockSum, m);
  __syncthreads();

  if (z == 0) atomicAdd(M_result_d, blockSum);
}

/*Returns an array of all the Boltzmann factors
  corresponding to different energies E<0.
  The factors are calculated beforehand to avoid
  calling exponential function on every iteration
  of the main loop*/
float *boltzmann_factors(float T, float J) {
  float *bf = malloc_float_h(6);

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
__global__ void update_spins(
  int n, int8_t *array, float *b_factors,
  float J, int which, pcg32_random_t *rng
) {

  int x, y, z, rng_idx;

  x = blockIdx.x;
  y = blockIdx.y;
  z = threadIdx.x*2 + (x + y + which) % 2;

  if (x >= n || y >= n || z >= n) {
    return;
  }

  rng_idx = index(x, y, z, n) / 2;

  float En = E(x, y, z, array, n, J);
  int b_idx = fabs(En/J)+0.1; //+0.1 to negotiate floating point inaccuracy

  float b_factor;
  if (b_idx > 0) { //to not call b_factors[-1]
    b_factor = b_factors[b_idx-1];
  } else {
    b_factor = 0;
  }

  if (En > 0 || rand_f(rng, rng_idx) < b_factor) {
    array[index(x,y,z,n)] *= -1;
  }
}

// A single simulation
float *run_simulation(
  int n, int mc_steps, int trans_steps, int calc, float T, float J,
  pcg32_random_t *rng_h, pcg32_random_t *rng_d
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

  dim3 grid_size(n, n, 1);
  dim3 block_half((int) ceil(n/2.0), 1, 1);
  dim3 block_whole(n, 1, 1);

  // Initializing spin array on host and copying it to the device
  int8_t *spins_h, *spins_d;
  spins_h = initial_array(n);
  malloc_int_d(&spins_d, n);
  cudaMemcpy(spins_d, spins_h, n*n*n, cudaMemcpyHostToDevice);

  // Precalculated boltzmann_factors
  float *b_factors_h, *b_factors_d;
  b_factors_h = boltzmann_factors(T,J);
  malloc_float_d(&b_factors_d, 6);
  cudaMemcpy(b_factors_d, b_factors_h, 6*sizeof(float), cudaMemcpyHostToDevice);

  int which = 0, samples = 0;

  //Transient mc steps
  for (int i = 0; i < trans_steps*2; i++) {
    // Change which squares to update every iteration
    which = (which + 1) % 2;

    // Update half of the spins
    update_spins<<<grid_size, block_half>>>(
      n, spins_d, b_factors_d, J, which, rng_d
    );
    cudaDeviceSynchronize();
  }

  float *E_h, *E_d, *M_h, *M_d;

  E_h = malloc_float_h(1);
  M_h = malloc_float_h(1);
  malloc_float_d(&E_d, 1);
  malloc_float_d(&M_d, 1);

  // Double precision used to avoid cumulating error from float summation
  double E_tot = 0, Mabs_tot = 0, E2_tot = 0, M2_tot = 0, M4_tot = 0;

  //Main loop where the averages are calculated
  for (int mc = 0; mc < mc_steps; mc++) {

    // Updating all spins
    for (int i = 0; i<2; i++) {

      // Change which squares to update every iteration
      which = (which + 1) % 2;

      update_spins<<<grid_size, block_half>>>(
        n, spins_d, b_factors_d, J, which, rng_d
      );
      cudaDeviceSynchronize();

    }

    // Calculating the physical quantities every calc mc steps
    if (mc % calc == 0) {

      // Calculating the current energy and magnetization
      *E_h = 0;
      *M_h = 0;
      cudaMemcpy(E_d, E_h, sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(M_d, M_h, sizeof(float), cudaMemcpyHostToDevice);

      E_total<<<grid_size, block_whole>>>(spins_d, n, J, E_d);
      cudaDeviceSynchronize();
      M_total<<<grid_size, block_whole>>>(spins_d, n, M_d);
      cudaDeviceSynchronize();

      cudaMemcpy(M_h, M_d, sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(E_h, E_d, sizeof(float), cudaMemcpyDeviceToHost);

      // Updating all the physical quantities on the host
      E_tot += *E_h/3;
      E2_tot += pow(*E_h/3,2);
      Mabs_tot += abs(*M_h);
      M2_tot += pow(*M_h, 2);
      M4_tot += pow(*M_h, 4);

      samples++;
    }

  }

  float norm = 1.0/(n*n*n*samples);
  float *ret = malloc_float_h(9);
  ret[0] = E_tot*norm; //mean energy
  ret[1] = E2_tot*norm; //mean energy^2
  ret[2] = Mabs_tot*norm; //mean |magnetization|
  ret[3] = M2_tot*norm; //mean magnetization^2
  ret[4] = M4_tot*norm; //mean magnetization^4
  ret[5] = (ret[1]-(ret[0]*ret[0]*n*n*n))/T; //heat capacity
  ret[6] = (ret[3]-(ret[2]*ret[2]*n*n*n))/T; //magnetic susceptibility
  ret[7] = 1-(ret[4]/(3*ret[3]*ret[3]*n*n*n)); //binder cumulant
  ret[8] = T; //temperature

  free(spins_h);
  free(b_factors_h);
  free(E_h);
  free(M_h);

  cudaFree(spins_d);
  cudaFree(b_factors_d);
  cudaFree(E_d);
  cudaFree(M_d);

  return ret;
}
