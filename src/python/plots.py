import numpy as np
import matplotlib.pyplot as plt

T = np.linspace(3,6,20);

for i in range(1,5):
    data = np.loadtxt("../../output/output" + str(i) + ".dat")
    plt.plot(T,data[:,2],'o-',label="$n=%d$"%(2**i))

plt.xlabel("$T$")
plt.ylabel(r"$\left\langle |m| \right\rangle $")
plt.legend()
plt.savefig("../../pics/mean_magnetization.png")
