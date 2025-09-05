from symmer.approximate import MPOOp
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_dmrg import DMRG2
import numpy as np
from symmer.operators import QuantumState, PauliwordOp
from typing import List

def get_MPO(operator: PauliwordOp, max_bond_dimension: int) -> MPOOp:
    """ Return the Matrix Product Operator (MPO) of a PauliwordOp 
    (linear combination of Paulis) given a maximum bond dimension
    """
    pstrings, coefflist = zip(*operator.to_dictionary.items())
    mpo = MPOOp(pstrings, coefflist, Dmax=max_bond_dimension)
    return mpo

def find_groundstate_quimb_verbose(
        MPOOp:      MPOOp, 
        dmrg:       DMRG2     = None, 
        gs_guess:   np.array  = None, 
        bond_dims:  List[int] = [8,16,32,64,128],
        max_sweeps: int       = 5
    ) -> QuantumState:
    """
    Use quimb's DMRG2 optimiser to approximate groundstate of MPOOp
    Args:
        MPOOp: MPOOp (MPO = Matrix Product Operator) representing operator
        dmrg: Quimb DMRG solver class
        gs_guess: Guess for the ground state, used as intialisation for the
                DMRG optimiser. Represented as a dense array.
        bond_dims: List of maximum bond dimensions per sweep of DMRG
        max_sweeps: Number of sweeps - the minimum value should be the length 
                of bond_dims, any number exceeding this will result in repetition 
                of the final element in bond_dims. 
    Returns:
        dmrg_state (QuantumState): Approximated groundstate
    """
    mpo = [np.squeeze(m) for m in MPOOp.mpo]
    MPO = MatrixProductOperator(mpo, 'dulr')

    if gs_guess is not None:
        no_qubits = int(np.log2(gs_guess.shape[0]))
        dims = [2] * no_qubits
        gs_guess = MatrixProductState.from_dense(gs_guess, dims)

    # Useful default for DMRG optimiser
    if dmrg is None:
        dmrg = DMRG2(MPO, bond_dims=bond_dims, cutoffs=1e-10, p0=gs_guess)
    dmrg.solve(verbosity=1, tol=1e-6, max_sweeps=max_sweeps)

    dmrg_state = dmrg.state.to_dense()
    dmrg_state = QuantumState.from_array(dmrg_state).cleanup(zero_threshold=1e-5)

    return dmrg_state

def overlap(state_1: QuantumState, state_2: QuantumState) -> float:
    """ Measures the norm of the overlap of two input QuantumStates
    """
    val = 0
    for bstring,left_coeff in state_1.to_dictionary.items():
        right_coeff = state_2.to_dictionary.get(bstring, 0)
        val += left_coeff.conjugate()*right_coeff
    return np.linalg.norm(val)