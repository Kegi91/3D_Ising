import numpy as np
import matplotlib.pyplot as plt

fpath = "../../output/benchmark/"
files = ["gtx1070.dat", "i7.dat"]

for fname in files:
    data = np.loadtxt(fpath + fname)
    size = data[:,0]
    t = data[:,1]

    plt.semilogy(size, t/size**3)

plt.savefig("../../pics/benchmark")
plt.show()
