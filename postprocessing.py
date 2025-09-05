import multiprocessing as mp
from itertools import product
import statsmodels.api as sm
import numpy as np
from copy import deepcopy
from symmer import PauliwordOp, QuantumState
from typing import List, Dict, Tuple, Callable

def bootstrap(
        operators: Dict[int, PauliwordOp], 
        states: Dict[int, QuantumState], 
        n_shots: int, 
        n_resamples: int = 100
    ) -> np.array:
    """ bootstrap the expectation values for some set of observables
    and corresponding measured states obtained from a quantum backend
    """
    cliquewise_resamples = []
    # loop over the observables and corresponding measurements
    for op,psi in zip(operators.values(), states.values()):
        # convert the measurement data into a normalized QuantumState
        state = QuantumState.from_dictionary(psi).normalize_counts
        resamples = []
        # resample the state with respect to the empirical measurement distribution
        # performs selection with replacement over the measurement data to generate
        # 'new' measurement sets to evaluate estimator variances
        for i in range(n_resamples):
            state_bootstrap = state.sample_state(n_shots).normalize_counts
            energy_resample = (state_bootstrap.dagger * op * state_bootstrap).real
            resamples.append(energy_resample)
        cliquewise_resamples.append(resamples)
    # returns an array of length n_resamples
    return np.sum(np.array(list(product(*cliquewise_resamples))), axis=1)

def bootstrap_noise_amp_k(
        data_in: dict, 
        diag_ops: Dict[int, PauliwordOp], 
        k: int, 
        n_shots: int
    ) -> List[Tuple[float, float]]:
    """ Given ZNE data, bootstrap each noise amplification factor, yielding
    a mean energy estimate and variance that we shall use for the purposes
    of weighted-least-squares regression in the extrapolation procedure.
    """
    bs_noise_factors = []
    for l in data_in['ZNE_factors']:
        bs_k = bootstrap(
            diag_ops, 
            data_in['energy_history']['data'][str(k)]['measurements'][str(l)],
            n_shots=n_shots
        )
        bs_noise_factors.append((np.mean(bs_k), np.var(bs_k)))
    return bs_noise_factors

def boostrap_zne(
        QWC_cliques: Dict[int, PauliwordOp], 
        data_in: dict, 
        n_shots: int
    ) -> List[List[Tuple[float, float]]]:
    """ Perform the bootstrapping procedure for each noise amplification factor in parallel
    """
    # assuming the circuits have had an appropriate change-of-basis, 
    # we correspondingly map the observable terms t Pauli Z's
    diag_ops = {
        clique_index:PauliwordOp(
            np.hstack([np.zeros_like(clique.X_block), clique.X_block | clique.Z_block]),
            clique.coeff_vec
        )
        for clique_index, clique in QWC_cliques.items()
    }
    # parallelize the bootstrapping across each noise amplification factor
    with mp.Pool(mp.cpu_count()) as pool:
        optimizer_hist_bs = pool.starmap(
            bootstrap_noise_amp_k, 
            [(data_in, diag_ops, k, n_shots) for k in range(len(data_in['energy_history']['values']))]
        )
    
    return np.asarray(optimizer_hist_bs)

def perform_wls_zne(
        QWC_cliques: Dict[int, PauliwordOp], 
        data_in: dict, 
        n_shots: int
    ) -> Tuple[float, float, Callable, np.array]:
    """ Having bootstrapped the ZNE data an obtained variances for each
    noise amplification factor, we now perfrm weighted-least-squares (WLS) regression
    with weights specified by 1/var(.) to penalize highly varying data points
    in the extrapolation procedure. We have found this to be more effective
    than ordinary-least-squares (OLS). This makes use of the statsmodels package.
    """
    Lambdas = data_in['ZNE_factors']
    optimizer_hist_bs = boostrap_zne(QWC_cliques, data_in, n_shots)
    
    X = sm.add_constant(Lambdas)

    zne_energy = []
    zne_stddev = []
    zne_wlsfnc = []
    # loop over optimiation steps
    for i in range(optimizer_hist_bs.shape[0]):
        # extract the noisy energy estimates and variances
        noisy_estimates, noisy_variances = zip(*optimizer_hist_bs[i])
        noisy_variances = np.array(noisy_variances)
        noisy_estimates = np.array(noisy_estimates)
        # initialize the linear regression driver from statsmodels.api
        wls = sm.WLS(endog=noisy_estimates, exog=X, weights=1/noisy_variances)
        # perform the WLS fitting procedure
        zne_wls_model = wls.fit()
        # this defines a regression curve, whose zero-value is our WLS-ZNE estimate
        zne_wls_curve = np.poly1d(zne_wls_model.params[::-1])
        zne_wls_estimate = zne_wls_model.params[0]
        # the variance from error propagation maybe be extracted 
        # from the standard error and residual degrees of freedom
        zne_wls_variance = np.square(zne_wls_model.HC0_se[0])*zne_wls_model.df_resid
        zne_wls_stddev = np.sqrt(zne_wls_variance)

        zne_energy.append(zne_wls_estimate)
        zne_stddev.append(zne_wls_stddev)
        zne_wlsfnc.append(zne_wls_curve)

    zne_energy = np.asarray(zne_energy)
    zne_stddev = np.asarray(zne_stddev)
    
    return zne_energy, zne_stddev, zne_wlsfnc, optimizer_hist_bs.transpose([2,0,1])

def symmetrize(
        data_in: dict, 
        symmetry: PauliwordOp, 
        permitted_value: int
    ) -> dict:
    """ Symmetry verification - discard measurements that violate the
    specified symmetry and assigned eigenvalue (defining the symmetry sector).
    """
    assert permitted_value in [+1,-1], 'Eigenvalue must be +/-1'
    # define the projector onto the symmetry sector (I + lambda*S)/2
    # where S is the symmetry and lambda its eigenvalue
    projector = (symmetry**0 + symmetry*permitted_value)*.5
    # make a copy of the dataset so it is not unintentionally malformed
    data_copy = deepcopy(data_in)
    ZNE_factors = list(map(str,data_copy['ZNE_factors']))
    opt_indices = list(data_copy['energy_history']['data'].keys())
    clq_indices = list(data_copy['QWC_cliques'].keys())
    # loop through all the measurement data in the input dataset and
    # force measurements to respect the given symmetry
    for i in opt_indices:
        for j in ZNE_factors:
            for k in clq_indices:
                measurements = QuantumState.from_dictionary(
                    data_copy['energy_history']['data'][i]['measurements'][j][k]
                ).normalize
                # apply the projector to the measured state, causing symmetry-violations to vanish
                rectified = (projector*measurements).normalize
                rectified_dict = dict(zip(rectified.to_dictionary.keys(), np.square(rectified.state_op.coeff_vec).real))
                data_copy['energy_history']['data'][i]['measurements'][j][k] = rectified_dict

    return data_copy