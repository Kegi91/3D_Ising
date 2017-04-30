This directory contains all the inputs needed by the main simulation.
The file input.dat is where all the inputs are read. It contains exmaple inputs, which can be run and plotted. The file critical_temps_input.dat contains the inputs
with which the Python 3 script cumulants.py can be run to obtain a high-accuracy 
approximation for the critical temperature of the model. 


The parameters needed are:

J = the coupling constant. J > 0 for ferromagnetic and J < 0 for antiferromagnetic model

T_min, T_max, T_len = The temperatures with which the simulation is run. They work 
similarly as the parameters of the function linspace() of Numpy.

sizes_min, sizes_max, step = The sizes of lattices with which the 
simulation is run. The lattice sizes run from sizes_min to sizes_max with the 
increment step.

mc, trans = Number of Monte Carlo and transient steps of each run.

fname = The name of the output file.


The parameters can be seperated by any white spaces. Other than that, the layout of
input.dat file should not be changed.
