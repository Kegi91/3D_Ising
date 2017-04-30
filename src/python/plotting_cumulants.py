import numpy as np
import matplotlib.pyplot as plt

def custom_label(i):
    if i == 1:
        return r'$U_{4}$'
    if i == 2:
        return r'$U_{8}$'
    if i == 3:
        return r'$U_{16}$'

def custom_color(i):
    if i == 1:
        return 'red'
    if i == 2:
        return 'blue'
    if i == 3:
        return 'green'

n = 4
fname = "cumulants"
T = np.loadtxt("../../output/" + fname + "1.dat")[:,8]

for i in range(1,n):
    data = np.loadtxt("../../output/" + fname + str(i) + ".dat")
    U = data[:,7]
    plt.plot(T,U,'o-',label=custom_label(i),color=custom_color(i))

plt.xlim(T[0], T[-1])
plt.legend(loc=3)
plt.xlabel("$T$")
plt.ylabel("$U_L$")
plt.title("Binder cumulants")
plt.grid(True)
plt.savefig("../../pics/binder_cumulants.png")
