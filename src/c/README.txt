This directory contains all the code of the main simulation and the makefile. 


The makefile contains the rules for:

make: Compile the code

make run: Compile and run the code

make clean: Remove the compiled file


The source code itself is split into different files for the sake of clarity:

main.c: Main program that starts the execution of the code

run.c: Handles reading the input file and writing the output file

ising.c: The implementation of the Metropolis algorithm to simulate the Ising model

utils.c: Utility functions e.g. memory allocation

pcg/pcg_basic.c: A powerful PRNG. Not coded by me. See pcg/ for a license
