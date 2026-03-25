import matplotlib
import scipy as sp
import numpy as np
import qutip
import math
import matplotlib.pyplot as plt

M_E = 511000
HBAR = 1240 / (2 * np.pi)
A = .75
V_0 = 15
N = 5000
L = 4 * A

def V(x, a): return V_0 if abs(x) > a else 0

def exp(x, a, b): return a * x ** b

#Part 1 Activity 1 Part 1

x = np.linspace(-L, L, N)
dx = x[2] - x[1]
x = x[1:-1]
t = HBAR**2 / (2 * M_E * dx**2)

h_d = np.zeros(len(x))
h_e = np.zeros(len(x) - 1)
for i in range(len(x)):
    h_d[i] = V(x[i], A) + 2 * t
    if i != len(x) - 1: h_e[i] = -t

ee, ef = sp.linalg.eigh_tridiagonal(h_d, h_e)
ee_filter = []
ef_filter = []
p = []

for v, f in zip(ee, ef.T):
    if v < V_0:
        ee_filter.append(v)
        ef_filter.append(f)

for f in ef_filter:
    a = sp.integrate.simpson(abs(f) * abs(f), x)
    p.append(f / np.sqrt(a))

for i, phi in enumerate(p):
    plt.figure()
    plt.plot(x, abs(phi**2), label=f'Eigenfunction {i+1}')
    plt.title(f'Eigenfunction {i+1} of the Hamiltonian')
    plt.xlabel('x')
    plt.ylabel('|ϕ(x)|^2f')
    plt.legend()
    plt.savefig(f'6.images/eigenfunctions{i+1}.png', dpi=300)
    plt.close()


q = [i for i in range(1, len(p))]
e = [ee_filter[i - 1] for i in q]
inside = (x >= -a) & (x <= a)
p1 = [sp.integrate.simpson(np.abs(p[i-1][inside])**2, x[inside]) for i in q]

df = pd.DataFrame({
    "Quantum Number (n)": q,
    "Eigenenergy (eV)": e,
    "Probability inside Well": p1
})

df.to_csv("waves1Activity1Plots/waves.csv", index=False)

# Part 1 Activity 2 Problem 1

all_q = [i for i in range(1, len(ee_filter) + 1)]
popt, pcov = sp.optimize.curve_fit(exp, all_q, ee_filter, p0 = [1, 2])
yfit = exp(all_q, *popt)

isw = [np.pi ** 2 * HBAR ** 2 * n **2 /(2 * M_E * (2 * A) ** 2) for n in all_q]

plt.figure()
plt.plot(all_q, ee_filter, label='Eigenenergies')
plt.plot(all_q, yfit, '-', label=f'Fit: E = {popt[0]:.3f} n^{popt[1]:.3f}')
plt.title(f'Eigenenergies vs Quantum Number with v0 = {V_0} eV')
plt.xlabel('Quantum Number (n)')
plt.ylabel('Eigenenergy (eV)')
plt.legend()
plt.savefig(f'6.images/eeVn_at_v0={V_0}.png', dpi=300)

# Part 1 Activity 1 Problem 2

widths = np.linspace(1, 10, 100)
grounde = []
firstee = []

for w in widths:
    a = w / 2
    l = 4 * a
    x = np.linspace(-l, l, N)
    dx = x[2] - x[1]
    x = x[1:-1]
    t = HBAR**2 / (2 * M_E * dx**2)

    h_d = np.zeros(len(x))
    h_e = np.zeros(len(x) - 1)
    for i in range(len(x)):
        h_d[i] = V(x[i], a) + 2 * t
        if i != len(x) - 1: h_e[i] = -t

    ee, ef = sp.linalg.eigh_tridiagonal(h_d, h_e)
    ee_filter = [i for i in ee if i < V_0]

    ee_filter = sorted(ee_filter)
    grounde.append(ee_filter[0])
    firstee.append(ee_filter[1])

popt1, pcov1 = sp.optimize.curve_fit(exp, widths, grounde, p0 = [1, -2])

plt.figure()
plt.plot(widths, grounde, 'bo', label='Ground State Energy(E0)')
plt.plot(widths, exp(widths, *popt1), 'b-', label=f'E0 Fit: B={popt1[1]:.2f}')
plt.xlabel('Well Width 2a (nm)')
plt.ylabel('Energy (eV)')
plt.legend()
plt.title('Ground State Energy vs Well Width')
plt.savefig('6.images/groundvwidth.png')

popt2, pcov2 = sp.optimize.curve_fit(exp, widths, firstee, p0 = [1, -2])

plt.figure()
plt.plot(widths, firstee, 'bo', label='First Excited State Energy(E1)')
plt.plot(widths, exp(widths, *popt2), 'b-', label=f'E1 Fit: B={popt2[1]:.2f}')
plt.xlabel('Well Width 2a (nm)')
plt.ylabel('Energy (eV)')
plt.legend()
plt.title('First Excited State Energy vs Well Width')
plt.savefig('6.images/firstevwidth.png')

# Part 1 Activity 3 Problem 1

EG = 1.74
ME = 511000
MEA = .13 * ME
MHA = .45 * ME
V_0E = V_0H = 4.0
RH_BAR = 197.326
HC = 1239.84

def VE(x, l): return V_0E if abs(x) > l else 0
def VH(x, l): return V_0H if abs(x) > l else 0

lengt = np.linspace(1, 10, 19)
lbd = []

for l in lengt:
    a = l / 2
    le = 10
    x = np.linspace(-le, le, N)
    dx = x[2] - x[1]
    x = x[1:-1]
    te = RH_BAR**2 / (2 * MEA * dx**2)

    h_d = np.zeros(len(x))
    h_e = np.zeros(len(x) - 1)
    for i in range(len(x)):
        h_d[i] = VE(x[i], a) + 2 * te
        if i != len(x) - 1: h_e[i] = -te

    ee, ef = sp.linalg.eigh_tridiagonal(h_d, h_e)
    ee_filter = [i for i in ee if i < V_0E]
    ee1 = ee_filter[0]

    th = RH_BAR**2 / (2 * MHA * dx**2)

    h_d = np.zeros(len(x))
    h_e = np.zeros(len(x) - 1)
    for i in range(len(x)):
        h_d[i] = VH(x[i], a) + 2 * th
        if i != len(x) - 1: h_e[i] = -th

    ee, ef = sp.linalg.eigh_tridiagonal(h_d, h_e)
    ee_filter = [i for i in ee if i < V_0H]
    eh1 = ee_filter[0]

    ep = EG + ee1 + eh1
    lbd.append(HC / ep)


plt.figure()
plt.plot(lengt, lbd, 'bo', label='')
plt.ylabel('Wavelength (nm)')
plt.xlabel('Quantum Dot Length (nm)')
plt.title('Quantum Dot Length vs. Wavelength')
plt.savefig('6.images/quantumdots.png')