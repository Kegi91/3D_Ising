import numpy as np
import matplotlib.pyplot as plt

fpath = "../../output/benchmark/"
files = ["gtx1070.dat", "gtx1080.dat", "k80.dat", "i7.dat", "i5.dat"]
labels = [
    "GeForce GTX 1070",
    "GeForce GTX 1080",
    "Tesla K80",
    "i7-7700K",
    "i5-4200U"
]

for i,fname in enumerate(files):
    data = np.loadtxt(fpath + fname)
    size = data[:,0]
    t = data[:,1]

    plt.xlabel("Lattice size (n)")
    plt.ylabel("Seconds per single spin flip")

    plt.semilogy(size, t/(size**3 * 1100), label=labels[i])
    plt.legend()

plt.savefig("../../pics/benchmark")
plt.show()
