import scipy as sp
import numpy as np
import qutip
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import curve_fit

# Part 1 Activity 1
HBAR = .6582
HBARSQ_2M = .0381
HBARW = .050
MWSQ = .0328
I = 1j

def V(x): return 0.5 * MWSQ * x**2

def classicalProb(x):
    A = 10                                      
    return 1/(np.pi * np.sqrt(A**2 - x**2))

def sinusoid(t, A, omega, phi, C):
    return A * np.cos(omega * t + phi) + C

points = np.linspace(-10, 10, 1500)
dx = points[1] - points[0]
points = points[1:-1]

t = HBARSQ_2M / (dx**2)

h_d = np.zeros(len(points))
h_e = np.zeros(len(points) - 1)
for i in range(len(points)):
    h_d[i] = V(points[i]) + 2 * t
    if i != len(points) - 1: h_e[i] = -t

ee, ef = sp.linalg.eigh_tridiagonal(h_d, h_e)
ef = ef.T
p = []

for f in ef:
    a = sp.integrate.simpson(f**2, points)
    p.append(f / np.sqrt(a))

'''for i, phi in enumerate(p):
    if i > 5 and i != 59: continue
    plt.figure()
    plt.plot(points, abs(phi**2), label=f'Eigenfunction {i+1}')
    plt.title(f'Eigenfunction {i+1} of the Hamiltonian')
    plt.xlabel('x')
    plt.ylabel('|ϕ(x)|^2')
    plt.savefig(f'6.images/efquadratic{i+1}.png', dpi=300)
    plt.close()

classresults = [classicalProb(x) for x in points]
plt.figure()
plt.plot(points, classresults, label = 'Classical Harmonic Oscillator Probability')
plt.title('Classical Harmonic Oscillator Probability')
plt.xlabel('x')
plt.ylabel('Probability')
plt.legend()
plt.savefig('6.images/classicalP.png', dpi=300)
plt.close()'''

# Part 2 Activity 2

X_0 = 2.0
SIGMA = .45
K_0 = 1.8
THRESHOLD = 1e-4

def Psi0(x): return (np.power(np.e, (I * K_0 * x) - ((x - X_0)**2 / (4 * SIGMA**2)))) / (np.power(2 * np.pi * SIGMA**2, .25))

psi00 = Psi0(points)
psir = np.zeros(len(points), dtype = complex)
cn = []
l = 0

while 1 - sum(abs(c)**2 for c in cn) > THRESHOLD:
    cl = sp.integrate.simpson(np.conjugate(p[l]) * psi00, points)
    cn.append(cl)
    psir += cl * p[l]
    l += 1

'''plt.figure()
plt.plot(points, abs(psi00**2), label='Actual Psi')
plt.title('Probability of Ψ')
plt.xlabel('x')
plt.ylabel('|Ψ(x)|^2')
plt.savefig(f'6.images/psi0.png', dpi=300)
plt.close()

plt.figure()
plt.plot(points, abs(psir**2), label='Reconstructed Psi')
plt.title(f'Probability of Reconstructed Ψ with {l + 1} terms')
plt.xlabel('x')
plt.ylabel('|Ψ_r(x)|^2')
plt.savefig(f'test.png', dpi=300)
plt.close()'''

# Part 2 Activity 3

T = np.linspace(0, 5 * 2 * np.pi * HBAR /  HBARW, 300) 

psi_frames = []
probability_frames = []

for frame in T:
    psi_t = np.zeros(len(points), dtype = complex)

    for n in range(len(cn)):
       psi_t += cn[n] * p[n] * np.exp((-I * ee[n] * frame) / HBAR)

    psi_frames.append(psi_t)

    prob_density = np.abs(psi_t)**2
    probability_frames.append(prob_density)

psi_frames = np.array(psi_frames)
probability_frames = np.array(probability_frames)

fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot(points, probability_frames[0], lw=2)

ax.set_xlabel("x (nm)")
ax.set_ylabel(r"$|\psi(x,t)|^2$")
ax.set_title("Time Evolution of Probability Density")

ax.set_xlim(points.min(), points.max())
ax.set_ylim(0, 1.1 * np.max(probability_frames))

time_text = ax.text(
    0.02, 0.95, "", transform = ax.transAxes,
    verticalalignment = "top"
)

def update(frame):
    line.set_ydata(probability_frames[frame])
    time_text.set_text(f"t = {T[frame]:.2f} fs")
    return line, time_text

anim = FuncAnimation(
    fig,
    update,
    frames = 300,
    interval = 40,
    blit = True
)

'''anim.save("6.images/sho_time_evolution.gif", writer = PillowWriter(fps=20))
plt.show()'''

# Part 2 Activity 4

OMEGA = 0.07596
x_expectation_vals = []

for prob in probability_frames:
    x_exp = sp.integrate.simpson(points * prob,points)
    x_expectation_vals.append(x_exp)

A_guess = (np.max(x_expectation_vals) - np.min(x_expectation_vals)) / 2
C_guess = np.mean(x_expectation_vals)
omega_guess = OMEGA 
phi_guess = 0

initial_guess = [A_guess, omega_guess, phi_guess, C_guess]

params, covariance = curve_fit(
    sinusoid,
    T,
    x_expectation_vals,
    p0 = initial_guess
)

A_fit, omega_fit, phi_fit, C_fit = params

print(f"Fitted amplitude A = {A_fit:.4f} nm")
print(f"Fitted angular frequency omega = {omega_fit:.4f} fs^-1")
print(f"Fitted phase phi = {phi_fit:.4f} rad")
print(f"Fitted offset C = {C_fit:.4f} nm")

print("\nExpected omega =", OMEGA)
print("Percent error =", abs((omega_fit - OMEGA) / OMEGA) * 100, "%")

t_fine = np.linspace(T.min(), T.max(), 1000)
fit_curve = sinusoid(t_fine, A_fit, omega_fit, phi_fit, C_fit)

plt.figure(figsize=(8,5))
plt.plot(T, x_expectation_vals, 'o', label="Data")
plt.plot(t_fine, fit_curve, '-', label="Fit")
plt.xlabel("time (fs)")
plt.ylabel(r"$\langle x \rangle (t)$ (nm)")
plt.title("Sinusoidal Fit to Expectation Value")
plt.legend()
plt.grid(True)
plt.savefig(f'6.images/<x>fit.png')