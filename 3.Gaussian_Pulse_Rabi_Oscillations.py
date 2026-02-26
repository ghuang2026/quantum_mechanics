import numpy as np
import matplotlib.pyplot as plt
import random
seed = 31                                 #for reproducability
random.seed(seed)
np.random.seed(seed)

from qutip import ( # type: ignore
    expect, basis, sigmax, sigmay, sigmaz,
    sesolve, Bloch, Options)

# Concatinates time, state, and expectation value lists after each pulse
def concat_data(result, times_all, states_all, expect_all):
    times_k = np.array(result.times)      # convert result.times to numpy array for t_offset

    # check for first pulse empty data
    if times_all is None:
        # first pulse: take everything as-is
        times_all  = times_k
        states_all = list(result.states)
        expect_all = [np.array(e) for e in result.expect]

    # for all pulses after first pulse
    else:
        # shift times of this pulse so it follows previous pulse
        t_offset  = times_all[-1]
        new_times = times_k[1:] + t_offset  # skip first to avoid duplicate

        # concatenate times
        times_all = np.concatenate([times_all, new_times])

        # concatenate states (skip first to avoid duplicate)
        states_all.extend(result.states[1:])

        # concatenate expectation values (each expect[j] is already an array)
        for j in range(len(expect_all)):
            expect_all[j] = np.concatenate(
                [expect_all[j], np.array(result.expect[j][1:])]
            )
    return times_all, states_all, expect_all

def getbasis(basislist):
    return basislist[np.random.randint(0, 3)]

def draw_jittered_args():
    return {"Omega0": np.random.normal(Args["Omega0"], Omega0_jitter),
            "sigma": np.random.normal(Args["sigma"], sigma_jitter),
            "t0": Args["t0"]}

def getBlochVector(state):
    return np.array([expect(Sx, state), expect(Sy, state), expect(Sz, state)])

def traceDistance(v1, v2):
    diff = np.subtract(v1, v2)
    return .5 * np.sqrt(diff.conj().T @ diff)

# ============================================================
# 1. Parameters
#    Work numerically with ħ = 1; H has units of rad/s.
# ============================================================

Omega0 = 2 * np.pi * 10e3   # 2π × 10 kHz ~ 6.283e4 rad/s

# Sigma in microseconds:
sigma_us = 10.0             # 10 microseconds
sigma    = sigma_us * 1e-6  # convert to seconds

# ============================================================
# 2. Time grid: positive-only times 0 → 10σ, pulse centered at 5σ
# ============================================================

t0 = 5 * sigma          # center the Gaussian at 5σ (in seconds)
t_start = 0.0
t_end   = 10 * sigma    # window [0, 10σ] in seconds

tlist    = np.linspace(t_start, t_end, 20)   # seconds
tlist_us = tlist * 1e6                        # microseconds for plotting

Args = {"Omega0": Omega0, "sigma": sigma, "t0": t0}          

# ============================================================
# 3. Time-dependent Hamiltonian H(t) = Omega_z(t) * Sz
# ============================================================

def Omega(t, args):
    """Z-rotation rate Omega_z(t) in rad/s (Gaussian pulse)."""
    Omega0 = args["Omega0"]
    sigma  = args["sigma"]
    t0     = args["t0"]
    return Omega0 * np.exp(-(t - t0)**2 / (2 * sigma**2))

# Spin operators
Sx = 0.5 * sigmax()
Sy = 0.5 * sigmay()
Sz = 0.5 * sigmaz()

# In QuTiP units, with ħ = 1, a Hamiltonian in rad/s is fine:
# H(t) = Omega(t) * Sz
#H = [[Sz, Omega]]

# ============================================================
# 4. Initial state and time evolution
#    Start in |+x> = (|0> + |1>)/sqrt(2).
# ============================================================

psi0 = (basis(2, 0) + basis(2, 1)).unit()
psi_ideal = psi0      #split into two vectors to track
psi_noisy = psi0

# Track expectations of sigma_x, sigma_y, sigma_z
e_ops = [2*Sx, 2*Sy, 2*Sz]  # gives <σx>, <σy>, <σz>

# Statistical Jitter
sigma_jitter_us = sigma_us * .005         # .5% error units microsecond
sigma_jitter    = sigma_jitter_us * 1e-6  # convert to seconds

Omega0_jitter = Omega0 * .001             # 0.1% error units rad/s

# Force QuTiP to store the states as well as expectations
opts = Options(store_states=True)

# list possible directions applied field
basislist=[Sx,Sy,Sz]

#Number of pulses
NumPulse = 50

#initialize arrays  *** BE CAREFULE WITH DATATYPES IN QuTip
#pulseNum = []           # list to store the index of pulses
#times_all = None        # 1D numpy array
#states_all = []         # list of Qobj (specal QuTip datatype)
#expect_all = None       # list of many numpy arrays

#initialize arrays  *** BE CAREFULE WITH DATATYPES IN QuTip
pulseNum = []             # list to store the index of pulses
times_all_ideal = None    # 1D numpy array; don't really need two here
times_all_noisy = None
states_all_ideal = []     # list of Qobj (specal QuTip datatype)
states_all_noisy = []
expect_all_ideal = None   # list of many numpy arrays
expect_all_noisy = None
blochvec_all_ideal = []   # list of many numpy arrays
blochvec_all_noisy = [] 
trace_all = []

for k in range(NumPulse):
    pulseNum.append(k+1)                      #get index of loop

    H_axis = getbasis(basislist)              #get random Hamiltonian
    H = [[H_axis,Omega]]                      #build Hamiltonian

    #ideal
    result_ideal = sesolve(H, psi_ideal, tlist, e_ops=e_ops, args=Args, options=opts)
    sxI_t, syI_t, szI_t = result_ideal.expect

    psi_ideal = result_ideal.states[-1]

    bloch_ideal = getBlochVector(psi_ideal)

    blochvec_all_ideal.append(bloch_ideal)
    times_all_ideal, states_all_ideal, expect_all_ideal = concat_data(result_ideal, times_all_ideal, states_all_ideal, expect_all_ideal)

    #jittered
    args_jitter = draw_jittered_args()

    # Solve Schrodinger Equation for current (k) pulse
    result_noisy = sesolve(H, psi_noisy, tlist, e_ops=e_ops, args=args_jitter, options=opts)
    sxN_t, syN_t, szN_t = result_noisy.expect

    # Set final state of sesolve() to be intial state for next loop
    psi_noisy = result_noisy.states[-1]

    bloch_noisy = getBlochVector(psi_noisy)

    blochvec_all_noisy.append(bloch_noisy)
    # Concatenation of data from each pulse into big lists
    times_all_noisy, states_all_noisy, expect_all_noisy = concat_data(result_noisy, times_all_noisy, states_all_noisy, expect_all_noisy)

    trace = traceDistance(bloch_ideal, bloch_noisy)
    trace_all.append(trace)

    plt.figure()
    plt.plot(tlist_us, szI_t, label=r'$\langle \sigma_z \rangle ideal$')
    plt.plot(tlist_us, szN_t, label=r'$\langle \sigma_z \rangle noisy$')
    plt.xlabel('time (µs)')
    plt.ylabel('expectation value')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('expectation value ' + str(k + 1) + '.png')

#result = sesolve(H, psi0, tlist, e_ops=e_ops, args=args, options=opts)
#sx_t, sy_t, sz_t = result.expect

# Optional: plot expectation values vs time in microseconds
#plt.figure()
#plt.plot(tlist_us, sx_t, label=r'$\langle \sigma_x \rangle$')
#plt.plot(tlist_us, sy_t, label=r'$\langle \sigma_y \rangle$')
#plt.plot(tlist_us, sz_t, label=r'$\langle \sigma_z \rangle$')
#plt.xlabel('time (µs)')
#plt.ylabel('expectation value')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()

plt.figure()
plt.plot(pulseNum, trace_all, label=r'trace distance')
plt.xlabel('pulse number')
plt.ylabel('maximum measurement error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('error vs pulse.png')
# ============================================================
# 5. Bloch sphere trajectory
# ============================================================

#b = Bloch()
#b.add_states(states_all)
#b.show()

plt.show()