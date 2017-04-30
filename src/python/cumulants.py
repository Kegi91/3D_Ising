import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import os

def f2(beta, x):
    return beta[0]*x**2 + beta[1]*x + beta[2]

def residual2(beta, y, x):
    return y - f2(beta,x)

def f3(beta,x):
    return beta[0]*x**3 + beta[1]*x**2 + beta[2]*x + beta[3]

def residual3(beta, y, x):
    return y - f3(beta,x)

def intersect(x,y):
    if x[0] > y[0]:
        for i in range(len(x)):
            if x[i] < y[i]:
                return i
    else:
        for i in range(len(x)):
            if x[i] > y[i]:
                return i

    return 0

def mean(x):
    sum = 0
    for i in range(len(x)):
        sum += x[i]

    return sum/len(x)

n = 2
fname = "run"
binder_cumulants = []
runs = 100

for j in range(runs):
    os.system("../c/main")

    if j == 0:
        T = np.loadtxt("../../output/run1.dat")[:,8]
        x = np.linspace(T[0],T[-1],10000)

    U = np.empty((n,len(T)))

    for i in range(1,n+1):
        data = np.loadtxt("../../output/" + fname + str(i) + ".dat")
        U[i-1] = data[:,7]
        fit, others = leastsq(residual2,np.ones(3),args=(U[i-1],T))

    U8 = U[0]
    U16 = U[1]

    fit8, others = leastsq(residual2,np.ones(3),args=(U8,T))
    fit16, others = leastsq(residual2,np.ones(3),args=(U16,T))

    # # Plotting the binder cumulants and the fits
    # # to cheack the behavior
    # plt.plot(T,U8,'o', color='blue',label='$U_8$')
    # plt.plot(T,U16,'o', color='green',label=r"$U_{16}$")
    # plt.plot(x,f2(fit8,x),color='blue',label=r'$U_{8}\:\mathrm{fit}$')
    # plt.plot(x,f2(fit16,x),color='green',label=r'$U_{16}\:\mathrm{fit}$')
    # plt.legend(loc=3)
    # plt.xlabel("$T$")
    # plt.ylabel("$U_L$")
    # plt.title("Binder cumulants")
    # plt.grid(True)
    # plt.show()

    T_c = x[intersect(f2(fit16,x),f2(fit8, x))]
    binder_cumulants.append(T_c)
    print("%d/%d\t%f"%(j+1,runs,T_c))

f = open("../../output/critical_temps.dat","w")
for i in range(len(binder_cumulants)):
    f.write(str(binder_cumulants[i]) + "\n")
f.close()
