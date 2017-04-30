This directory contains most of the data analysis scripts used. 


plots.py:
Plots the magnetization, energy, heat capacity and magnetic susceptibility as a
function of the temperature. The file name from which the results are read can
be adjusted on line 16 and number of the lattices run on line 17. One might also
want to adjust the function custom_label(i) to correspond to the lattice sizes that
were run, or omit the legend.

plotting_cumulants.py:
Plots the fourth order Binder cumulants of the previous run. By default it plots
from the files cumulantsX.dat, which showcaseas the critical point. Last run can
be plotted by changing the line 21 to: 'fname = "run"' and line 20 to correspond to
the number of lattices that were run. Again, the labels and colors may need to be 
adjusted. 
The Binder cumulants are plotted in a different file than the other physical
quantities since they should be calculated really close to the critical temperature 
to gather meaningful data.

cumulants.py:
Runs n simulations and saves the temperatures, where the Binder cumulants intersect 
i.e. the critical temperatures, in the file ../../output/critical_temps.dat. 
A high-accuracy approximation for the critical temperature can be obtained by using
the inputs provided in the file ../../input/critical_temps_input.dat. Running this
with Intel i5-4200U @ 1.60GHz takes ~6 hours.
