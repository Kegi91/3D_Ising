# Three-dimensional Ising model

This is a simulation of the three-dimensional Ising model. It uses the Metropolis
algorithm to find the spin configuration, which minimizes the free energy of the
system. Implementation exists both for CPU and GPU. 

Example run on CPU can be executed and the results viewed from the directory src/c/ with
the following commands:
```
make run
python3 ../python/plots.py
eog ../../pics
```

Running on the GPU requires a Nvidia graphics card with minimum 3.0 compute capability
and CUDA Toolkit 5.0 or newer. Example run on GPU can be executed and the results viewed 
from the directory src/cuda/ with the following commands:
```
make run    
python3 ../python/plots.py
```

Most of the directories contain also their own README.txt files to further explain
this project.
