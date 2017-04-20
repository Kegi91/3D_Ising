import numpy as np
import matplotlib.pyplot as plt

def custom_label(i):
    if i == 1:
        return "$n=20$"
    elif i == 2:
        return "$n=25$"
    elif i == 3:
        return "$n=30$"
    elif i == 4:
        return "$n=40$"

T = np.linspace(1,8,20);
n = 4

for i in range(1,n+1):
    data = np.loadtxt("../../output/output" + str(i) + ".dat")
    plt.plot(T,data[:,2],'o-',label=custom_label(i))

plt.xlabel("$T$")
plt.ylabel(r"$\left\langle |m| \right\rangle $")
plt.legend()
plt.title("Mean magnetization")
plt.savefig("../../pics/mean_magnetization.png")
plt.clf()

for i in range(1,n+1):
    data = np.loadtxt("../../output/output" + str(i) + ".dat")
    plt.plot(T,data[:,0],'o-',label=custom_label(i))

plt.xlabel("$T$")
plt.ylabel(r"$\left\langle E \right\rangle $")
plt.legend(loc=4)
plt.title("Mean energy")
plt.savefig("../../pics/mean_energy.png")
plt.clf()

for i in range(1,n+1):
    data = np.loadtxt("../../output/output" + str(i) + ".dat")
    C = data[:,5]
    plt.plot(T,C,'o-',label=custom_label(i))

plt.xlabel("$T$")
plt.ylabel(r"$c_V$")
plt.legend()
plt.title("Heat capacity")
plt.savefig("../../pics/heat_capacity.png")
plt.clf()

for i in range(1,n+1):
    data = np.loadtxt("../../output/output" + str(i) + ".dat")
    X = data[:,6]
    plt.plot(T,X,'o-',label=custom_label(i))

plt.xlabel("$T$")
plt.ylabel(r"$\chi$")
plt.legend()
plt.title("Magnetic susceptibility")
plt.savefig("../../pics/susceptibility.png")
plt.clf()
