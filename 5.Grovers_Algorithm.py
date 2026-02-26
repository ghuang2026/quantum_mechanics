import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import ZGate

n_comp = 6
n_anc  = 13
N = n_comp + n_anc

comp = list(range(n_comp))
anc  = list(range(n_comp, N))

MCZ = ZGate().control(n_comp - 1)

backend = Aer.get_backend("aer_simulator")
shots = 4000

def oracle_gate(n_comp: int, n_anc: int):
    n_total = n_comp + n_anc
    qc_oracle = QuantumCircuit(n_total)
    #1
    #qc_oracle.barrier()
    qc_oracle.x(0)
    qc_oracle.cx(0, 6)
    qc_oracle.x(0)
    #qc_oracle.barrier()
    #2
    qc_oracle.x(0)
    qc_oracle.ccx(0, 5, 7)
    qc_oracle.x(0)
    #qc_oracle.barrier()
    #3
    qc_oracle.x(0)
    qc_oracle.x(4)
    qc_oracle.ccx(0, 4, 8)
    qc_oracle.x(4)
    qc_oracle.x(0)
    #qc_oracle.barrier()
    #4
    qc_oracle.x(4)
    qc_oracle.ccx(0, 4, 9)
    qc_oracle.x(4)
    #qc_oracle.barrier()
    #5
    qc_oracle.x(2)
    qc_oracle.x(3)
    qc_oracle.ccx(2, 3, 10)
    qc_oracle.x(3)
    qc_oracle.x(2)
    #qc_oracle.barrier()
    #6
    qc_oracle.x(2)
    qc_oracle.ccx(2, 3, 11)
    qc_oracle.x(2)
    #qc_oracle.barrier()
    #7
    qc_oracle.x(2)
    qc_oracle.ccx(1, 2, 12)
    qc_oracle.x(2)
    #qc_oracle.barrier()
    #8
    qc_oracle.ccx(1, 2, 13)
    #qc_oracle.barrier()
    #9
    qc_oracle.x(4)
    qc_oracle.ccx(3, 4, 14)
    qc_oracle.x(4)
    #qc_oracle.barrier()
    #10
    qc_oracle.ccx(3, 4, 15)
    #qc_oracle.barrier()
    #11
    qc_oracle.x(2)
    qc_oracle.x(4)
    qc_oracle.mcx([0, 2, 4], 16)
    qc_oracle.x(4)
    qc_oracle.x(2)
    #qc_oracle.barrier()
    #12
    qc_oracle.x(0)
    qc_oracle.x(4)
    qc_oracle.mcx([0, 3, 4], 17)
    qc_oracle.x(4)
    qc_oracle.x(0)
    #qc_oracle.barrier()
    #Eval
    for i in anc: qc_oracle.x(i)
    qc_oracle.mcx(anc[:-1], 18)
    for i in anc: qc_oracle.x(i)
    #Flip
    #qc_oracle.barrier()
    qc_oracle.z(18)
    #qc_oracle.barrier()
    #Eval
    for i in anc: qc_oracle.x(i)
    qc_oracle.mcx(anc[:-1], 18)
    for i in anc: qc_oracle.x(i)
    #12
    qc_oracle.x(0)
    qc_oracle.x(4)
    qc_oracle.mcx([0, 3, 4], 17)
    qc_oracle.x(4)
    qc_oracle.x(0)
    #qc_oracle.barrier()
    #11
    qc_oracle.x(2)
    qc_oracle.x(4)
    qc_oracle.mcx([0, 2, 4], 16)
    qc_oracle.x(4)
    qc_oracle.x(2)
    #qc_oracle.barrier()
    #10
    qc_oracle.ccx(3, 4, 15)
    #qc_oracle.barrier()
    #9
    qc_oracle.x(4)
    qc_oracle.ccx(3, 4, 14)
    qc_oracle.x(4)
    #qc_oracle.barrier()
    #8
    qc_oracle.ccx(1, 2, 13)
    #qc_oracle.barrier()
    #7
    qc_oracle.x(2)
    qc_oracle.ccx(1, 2, 12)
    qc_oracle.x(2)
    #qc_oracle.barrier()
    #6
    qc_oracle.x(2)
    qc_oracle.ccx(2, 3, 11)
    qc_oracle.x(2)
    #qc_oracle.barrier()
    #5
    qc_oracle.x(2)
    qc_oracle.x(3)
    qc_oracle.ccx(2, 3, 10)
    qc_oracle.x(3)
    qc_oracle.x(2)
    #qc_oracle.barrier()
    #4
    qc_oracle.x(4)
    qc_oracle.ccx(0, 4, 9)
    qc_oracle.x(4)
    #qc_oracle.barrier()
    #3
    qc_oracle.x(0)
    qc_oracle.x(4)
    qc_oracle.ccx(0, 4, 8)
    qc_oracle.x(4)
    qc_oracle.x(0)
    #qc_oracle.barrier()
    #2
    qc_oracle.x(0)
    qc_oracle.ccx(0, 5, 7)
    qc_oracle.x(0)
    #qc_oracle.barrier()
    #1
    qc_oracle.x(0)
    qc_oracle.cx(0, 6)
    qc_oracle.x(0)
    #qc_oracle.barrier()

    return qc_oracle.to_gate(label="Oracle")

def diffuser_gate(n_comp: int):
    qc_diff = QuantumCircuit(n_comp)
    comp = range(n_comp)
    qc_diff.h(comp)
    qc_diff.x(comp)         
    qc_diff.append(MCZ, comp) 
    qc_diff.x(comp)
    qc_diff.h(comp)
    return qc_diff.to_gate(label="Diffuser")

qc = QuantumCircuit(N, n_comp)
O = oracle_gate(n_comp, n_anc)
D = diffuser_gate(n_comp)

qc.h(comp)

qc.barrier()

for i in range(int(np.pi / 4 * np.sqrt(np.pow(2, n_comp)))):
    qc.append(O, comp + anc)
    qc.append(D, comp)

qc.barrier()

qc.measure(comp, list(range(n_comp)))

qc_run = transpile(qc, backend, optimization_level=0)

result = backend.run(qc_run, shots=shots).result()
counts = result.get_counts()

fig = plot_histogram(counts)
plt.savefig("histprom.png")

fig = qc.draw("mpl")
fig.suptitle(r"Grover Circuit for Prom Problem", fontsize=14)
plt.savefig("cktprom.png")