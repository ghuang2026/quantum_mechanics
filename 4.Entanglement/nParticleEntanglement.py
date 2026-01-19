import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from qutip import ( # pyright: ignore[reportMissingImports]
    basis, tensor, qeye, ket2dm, ptrace,
    entropy_vn
)

# QuTip puts purity as an attribute of the density matrix.
# You can use it as rho.purity()
# or you can define a specific function to extract the purity.

def purity(rho):    
    return rho.purity()

np.set_printoptions(precision=4, suppress=True)