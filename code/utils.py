# Added to silence some warnings.
from jax.config import config
config.update("jax_enable_x64", True)

import jax
import flax
import optax
import jax.numpy as jnp

import pennylane as qml
import netket as nk


import numpy as np
from functools import partial, reduce



##########################################################################################################################
## Utils for circuit evaluation - PENNYLANE

def e_i(n,i):
    '''
    This function returns the i-th vector of the canonical basis of R^n.
    '''
    vi = np.zeros(n)
    vi[i] = 1.0
    return jnp.array(vi)


partial(jax.jit, static_argnames=("hi"))
def validate_samples(hi, v, mel):
    '''
    This functions takes a batch of samples and verify they are valid according to the hilbert space.

    Args:
        hi:  NetKet Hilbert space object
        v:   batch of samples
        mel: batch of classic matrix elements
    '''
    mask = jnp.full(v.shape[:-1], True)
    for i,h in enumerate(hi._hilbert_spaces):
        i_start = hi._cum_indices[i]
        i_end = hi._cum_indices[i+1]
        v_i = ((v[..., i_start:i_end] +1)/2).sum(axis=-1)
        mask_i = jnp.logical_and(v_i >= h.minimum_up, v_i <= h.maximum_up)    
        mask = jnp.logical_and(mask, mask_i)

    if v.ndim == 2:
        v0 = v[0]
        mask_r = mask
    elif v.ndim == 3:
        v0 = v[0,0]
    mask_r = np.reshape(mask, mask.shape + (-1,))
    v = mask_r * v + np.logical_not(mask_r) * v0
    return v, mel * mask


#@qml.qnode(device_with_ancilla, interface="jax")
def general_hadamard_test(dev, circuit, params, θ_σ, θ_η, operator, part=None):
    '''
    This function implements the general Hadamard test for estimating the overlap between two states in the hybrid algorithm.

    Args:
        dev:      PennyLane device
        circuit:  Pennylane circuit to be created on the device
        params:   parameters of the circuit (they are the same for both the states)
        θ_σ:      parameters of the classical input for the first state
        θ_η:      parameters of the classical input for the second state
        operator: operator to be measured
        part:     part of the overlap to be estimated. It can be "Re" or "Im". If None, the real part is estimated.
    '''
    
    dev_qubits = dev.num_wires

    @qml.qnode(dev, interface="jax")
    def _general_hadamard_test():
    
        circ_left, circ_classical_input, circ_right = circuit 
    

        # First Hadamard gate applied to the ancillary qubit.
        qml.Hadamard(wires="a")

        # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
        # phase gate.
        if part == "Im" or part == "im":
            qml.PhaseShift(-np.pi / 2, wires="a")

        circ_left(dev, params)
        circ_classical_input(dev, params, θ_σ,θ_η)
        circ_right(dev, params)

        # Second Hadamard gate applied to the ancillary qubit.
        qml.Hadamard(wires="a")

        # Expectation value of Z for the ancillary qubit.
        op = operator(wires=range(dev_qubits-1)) @ qml.PauliZ(wires="a")

        return qml.expval(op)
    
    return _general_hadamard_test()


def overlap_estimation(dev, circuit, params, θ_σ, θ_η, operator,complex_val=True):
    '''
    This function returns the overlap between two states in the hybrid algorithm.

    Args:
        dev:         PennyLane device
        circuit:     Pennylane circuit to be created on the device
        params:      parameters of the circuit (they are the same for both the states)
        θ_σ:         parameters of the classical input for the first state
        θ_η:         parameters of the classical input for the second state
        operator:    operator to be measured
        complex_val: if True, the whole complex overlap is returned. If False, the real part is returned.
    '''

    ovp_real = general_hadamard_test(dev, circuit, params, θ_σ, θ_η, operator, part="Re")

    if not complex_val:
        return ovp_real
    else:
        ovp_img = general_hadamard_test(dev, circuit, params, θ_σ, θ_η, operator, part="Im")
        return ovp_real+ 1.0j * ovp_img

def circuit_state(dev, circuit, params, θ_σ):
    '''
    This function returns the state of a circuit.
    
    Args:
        dev:      PennyLane device
        circuit:  Pennylane circuit to be created on the device
        params:   parameters of the circuit
        θ_σ:      parameters of the classical input

    '''

    circ_left, circ_classical_input, circ_right = circuit 
    
    @qml.qnode(dev, interface="jax")
    def _circuit_state():
        
        circ_left(dev, params)
        circ_classical_input(dev, params, θ_σ, None)
        circ_right(dev, params)
        
        return qml.state()

    return _circuit_state()

#@qml.qnode(device_phys, interface="jax")
def expect_value(dev, circuit, params, θ_σ, op):
    '''
    This function returns the expectation value of an operator on the quantum circuit.

    Args:
        dev:      PennyLane device
        circuit:  Pennylane circuit to be created on the device
        params:   parameters of the circuit
        θ_σ:      parameters of the classical input
        op:       operator to be measured
    '''
    
    dev_qubits = dev.num_wires
    
    @qml.qnode(dev, interface="jax")
    def _expect_value():
        
        circ_left, circ_classical_input, circ_right = circuit 
    
        circ_left(dev, params)
        circ_classical_input(dev, params, θ_σ, None)
        circ_right(dev, params)
    
        return qml.expval(op(wires=range(dev_qubits)))
        
    return _expect_value()


def evaluate_ansatz(ma, dev, circuit, sample_to_angle, hilbert, params):
    '''
    This function evaluates the ansatz on the quantum device.
    Evaluating the wavefunction is exponentially expensive in the size of the system.

    Args:
        ma:              Netket classic model
        dev:             PennyLane device
        circuit:         Pennylane circuit to be created on the device
        sample_to_angle: function that maps the classical input to the quantum circuit
        hilbert:         Hilbert space of the classic model
        params:          parameters of the hybrid ansatz circuit
    '''
    
    params_ac, params_q = params.pop("quantum")
    params_c, params_a = params_ac.pop("angles")
    
    # Classical wfn
    σ  = hilbert.all_states()
    wfn_c = jnp.exp(ma.apply(params_c, σ))
    wfn_c = wfn_c / jnp.sqrt(jnp.sum(jnp.abs(wfn_c)**2))
    
    # Quantum wfn
    θ_σ   = sample_to_angle.apply(params_a,σ)
    wfn_q = jax.vmap(circuit_state,in_axes=(None, None, None, 0), out_axes = 0)(dev, circuit, params_q, θ_σ)
    
    # Construct total wfn
    wfn_tot = 0
    σ_n = hilbert.states_to_numbers(σ)
    for i in σ_n:
        v = wfn_c[i]*e_i(σ_n.shape[0],i)
        
        wfn_tot = wfn_tot + jnp.kron(wfn_q[i],v)
        
    
    return wfn_tot 



##########################################################################################################################
##########################################################################################################################
## NETKET

# Create a Fake Variational State to use SR on the classical model
class FakeVariationalState:
    def __init__(self, model, parameters, samples):
        self._apply_fun = model.apply
        self.parameters = parameters
        self.model_state = dict()
        self.samples = samples



def e_loc(ma, dev, circ, pars, σ, σp, mels, sample_to_angle, op_q):
    '''
    This function returns the local energy element for each classic sample and a given operator op = op_q ⊗ op_c

    Args:
        ma:              NetKet model
        dev:             Pennylane device
        circ:            Pennylane circuit
        pars:            Parameters of the circuit
        σ:               Classic samples
        σp:              Classic samples connected to σ
        mels:            Matrix connected elements op_c^{σ,σp}
        sample_to_angle: Function that maps the classic samples to the quantum angles
        op_q:            Quantum part of the operator
        
    '''

    # Divide the parameters
    pars_ac, pars_quantum = pars.pop('quantum')
    pars_c, pars_a        = pars_ac.pop('angles')
    # Compute the classical local terms
    local_terms_classical =  mels *jnp.exp(ma.apply(pars_c, σp) - ma.apply(pars_c, σ))
    
    # Compute the quantum local terms
    θ_σ = sample_to_angle.apply(pars_a,σ)
    θ_σp = jax.vmap(lambda η: sample_to_angle.apply(pars_a,η))(σp)
    local_terms_quantum = jax.vmap(lambda θ_η: overlap_estimation(dev, circ, pars_quantum, θ_σ, θ_η, op_q, complex_val=True))(θ_σp)
    
    return jnp.sum(local_terms_classical * local_terms_quantum, axis=-1)




#========================================================================================================================
def e_loc_and_grad(ma, dev, circ, pars, σ, σp, mels, sample_to_angle, op_q):

    '''
    Function to compute the local energy and the gradient of the local energy for each classic sample and a given operator op = op_q ⊗ op_c

    Args:
        ma:              NetKet model
        dev:             Pennylane device
        circ:            Pennylane circuit
        pars:            Parameters of the hybrid model
        σ:               Classic samples
        σp:              Classic samples connected to σ
        mels:            Matrix connected elements op_c^{σ,σp}
        sample_to_angle: Function that maps the classic samples to the quantum angles
        op_q:            Quantum part of the operator
    '''
    pars_ac, pars_quantum = pars.pop('quantum')
    pars_c, pars_a       = pars_ac.pop('angles')
    
    local_terms_classical =  mels * jnp.exp(ma.apply(pars_c, σp) - ma.apply(pars_c, σ))
    
    # Compute the "angle" local terms
    
    local_terms_angle_fun      = jax.vmap(lambda w_pars, wq_pars, η: overlap_estimation(dev, circ, wq_pars, sample_to_angle.apply(w_pars,σ), sample_to_angle.apply(w_pars,η), op_q, complex_val=True), in_axes=(None, None, 0), out_axes=0)
    local_terms_angle, vjp_fun = nk.jax.vjp(lambda w, wq: local_terms_angle_fun(w, wq, σp) , pars_a, pars_quantum)

    
    e_loc  = jnp.sum(local_terms_classical * local_terms_angle, axis=-1)
    grad_a, grad_q = vjp_fun(local_terms_classical)


    ### Standard estimator

    log_term_fun = jax.vmap(lambda wc_pars, η: ma.apply(wc_pars,η),in_axes=(None, 0), out_axes=0)
    log_terms, log_vjp_fun = nk.jax.vjp(lambda wc: log_term_fun(wc, σp), pars_c)
    q_mels = local_terms_classical*jnp.real(local_terms_angle)
    f_c = log_vjp_fun(q_mels)[0]

    
    return e_loc, grad_a, grad_q, f_c


def e_loc_and_grad_unbiased(ma, ma_lin, dev, circ, pars, σ, σp, mels, sample_to_angle, op_q):
    '''
    Function to compute the local energy and the gradient of the local energy for each classic sample and a given operator op = op_q ⊗ op_c.
    It used an unbiased version of the estimator
    Args:
        ma:              NetKet model
        ma_lin:          Linearized NetKet model
        dev:             Pennylane device
        circ:            Pennylane circuit
        pars:            Parameters of the hybrid model
        σ:               Classic samples
        σp:              Classic samples connected to σ
        mels:            Matrix connected elements op_c^{σ,σp}
        sample_to_angle: Function that maps the classic samples to the quantum angles
        op_q:            Quantum part of the operator
    '''

    pars_ac, pars_quantum = pars.pop('quantum')
    pars_c, pars_a       = pars_ac.pop('angles')
    
    local_terms_classical =  mels * jnp.exp(ma.apply(pars_c, σp) - ma.apply(pars_c, σ))
    
    # Compute the "angle" local terms
    
    local_terms_angle_fun      = jax.vmap(lambda w_pars, wq_pars, η: overlap_estimation(dev, circ, wq_pars, sample_to_angle.apply(w_pars,σ), sample_to_angle.apply(w_pars,η), op_q, complex_val=True), in_axes=(None, None, 0), out_axes=0)
    local_terms_angle, vjp_fun = nk.jax.vjp(lambda w, wq: local_terms_angle_fun(w, wq, σp) , pars_a, pars_quantum)

    #print("local terms angle:",local_terms_angle.shape)
    e_loc  = jnp.sum(local_terms_classical * local_terms_angle, axis=-1)
    grad_a, grad_q = vjp_fun(local_terms_classical)


    # Unbiased estimator

    ## Linear
    psi = ma_lin.apply(pars_c, σ)
    ## Exp(log)
    #psi = jnp.exp(ma.apply(pars_c, σ))

    ## Linear
    lin_term_fun = jax.vmap(lambda wc_pars, η: ma_lin.apply(wc_pars,η),in_axes=(None, 0), out_axes=0)
    ## Exp(log)
    #lin_term_fun = jax.vmap(lambda wc_pars, η: jnp.exp(ma.apply(wc_pars,η)),in_axes=(None, 0), out_axes=0)

    lin_terms, lin_vjp_fun = nk.jax.vjp(lambda wc: lin_term_fun(wc, σp), pars_c)
    q_mels = jnp.real(local_terms_angle* mels)

    # This gives the unbiased estimator of the classic gradient
    f_c = lin_vjp_fun(q_mels)[0]
    f_c = jax.tree_map(lambda x: x/psi,f_c)
    
    return e_loc, grad_a, grad_q, f_c


# Vmap on multiple samples and jax.jit
e_loc_batched                   = jax.jit(jax.vmap(e_loc           , in_axes=(None, None, None, None, 0, 0, 0, None, None), out_axes=0),     static_argnums=(0, 1, 2, 7, 8))
e_loc_and_grad_batched          = jax.jit(jax.vmap(e_loc_and_grad, in_axes=(None, None, None, None, 0, 0, 0, None, None), out_axes=(0,0,0,0)),static_argnums=(0, 1, 2, 7, 8))
e_loc_and_grad_unbiased_batched = jax.jit(jax.vmap(e_loc_and_grad_unbiased, in_axes=(None, None, None, None, None, 0, 0, 0, None, None), out_axes=(0,0,0,0)),static_argnums=(0, 1, 2, 3, 8, 9))



##### Use the jitted functions to compute the energy and derivatives



def e_locs_total(ma, dev, circ, pars, σ_batch, sample_to_angle, h_mixed,valid_samples=False):
    
    '''
    This function takes in input a batch of samples
    and computes the sum of e_locs for each term of the hamiltonian

    Args:
        ma:              Netket model
        dev:             Pennylane device
        circ:            Pennylane circuit
        pars:            total parameters of the hybrid model
        σ_batch:         batch of classic samples
        sample_to_angle: function to convert classic samples to angles in the quantum circuit
        h_mixed:         list of tuples (hq,hc) where hq is the quantum part of the hamiltonian and hc is the classical part
        valid_samples:   if True, the samples are validated wrt particle conservation before computing the energy
    '''
    
    e_locs = 0
    
    for (hq,hc) in h_mixed:
        σp, mels = hc.get_conn_padded(σ_batch)
        if valid_samples:
            σp, mels = validate_samples(hc.hilbert, σp, mels) 
        e_locs   = e_locs + e_loc_batched(ma, dev, circ, pars, σ_batch, σp, mels, sample_to_angle, hq)
    return e_locs

def e_tot(ma, dev, circ, pars, σ_batch, sample_to_angle, h_mixed,valid_samples=False):
    '''
    Computes the sum of the local energies

    Args:
        ma:              Netket model
        dev:             Pennylane device
        circ:            Pennylane circuit
        pars:            total parameters of the hybrid model
        σ_batch:         batch of classic samples
        sample_to_angle: function to convert classic samples to angles in the quantum circuit
        h_mixed:         list of tuples (hq,hc) where hq is the quantum part of the hamiltonian and hc is the classical part
        valid_samples:   if True, the samples are validated wrt particle conservation before computing the energy
    '''


    e_locs = e_locs_total(ma, dev, circ, pars, σ_batch, sample_to_angle, h_mixed,valid_samples)
    
    return nk.stats.statistics(e_locs.T)

## Optimization utils

@partial(jax.jit, static_argnums=(0))
def compute_Ok(ma, pars, σ_batch):
    '''
    Computes the Ok matrix

    Args:
        ma:       Netket model
        pars:     parameters of the classic model
        σ_batch:  batch of classic samples
    '''

    logpsi, Ok_fun = nk.jax.vjp(lambda w: ma.apply(w, σ_batch), pars)
    Ok              = Ok_fun(jnp.ones_like(logpsi))[0]
    #grad            = vjp_fun(delta_e_loc.reshape(-1)/σ_batch.shape[0])
    Ok = jax.tree_map(lambda x:x/σ_batch.shape[0], Ok)
    
    return Ok

# Define a function to compute SR on the classical params
def compute_SR(ma, sigma, grad, pars_c, diag_shift=0.001):
        '''
        Computes the SR on the classical parameters

        Args:
            ma:          Netket model
            sigma:       batch of classic samples
            grad:        Gradient of the energy
            pars_c:      Classical parameters
            diag_shift:  Shift to add to the diagonal of the S matrix to stabilize the cholesky decomposition
        '''
        
        # Grad should be real

        grad_ac, grad_q = grad.pop("quantum")
        grad_c, grad_a  = grad_ac.pop("angles")


        pars_c_mod = pars_c.pop("params")[1]
        grad_c     = grad_c.pop("params")[1]
        
        # Compute the S Matrix
        fake_vs  = FakeVariationalState(ma, pars_c_mod, sigma)
        S        = nk.optimizer.qgt.QGTJacobianPyTree(fake_vs,diag_shift=diag_shift)


        
        grad_mod, _ = S.solve(nk.optimizer.solver.cholesky, grad_c)

        grad_all = {}
        grad_all['params'] = grad_mod
        grad_all['angles']  = grad_a
        grad_all['quantum'] = grad_q

        grad_final = flax.core.freeze(grad_all)

        return grad_final

def optimizer_step(optimizer,opt_state, grads, params):
    '''
    Performs a single optimization step using optax

    Args:
        optimizer: optax optimizer
        opt_state: optax optimizer state
        grads:     gradients for the classic and quantum parameters
        params:    parameters for the classic and quantum models
    '''
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

## Final function to compute energy and gradient

def e_tot_and_grad(ma, ma_lin, dev, circ, pars, σ, sample_to_angle, h_mixed, unbiased=False, valid_samples=False):
    
    '''
    This function will return the energy  and its gradient

    Args:
        ma:              NetKet classic model
        ma_lin:          NetKet linear model (for the unbiased estimator)
        dev:             pennylane device
        circ:            pennylane circuit
        pars:            parameters of the model
        σ:               classic samples
        sample_to_angle: function to convert classic samples to angles in the quantum circuit
        h_mixed:         Hamiltonian in the mixed form
        unbiased:        if True, the unbiased estimator will be used
        valid_samples:   if True, the samples will be validated for particle preserving ansatzes
    
    '''
    
    # with a similar structure to those of parameters, so having grad['quantum']
    
    assert σ.ndim == 3
    n_chains = σ.shape[1]
    σ_batch = σ.reshape(-1, σ.shape[-1])
    
    # intialise quantities
    e_locs = 0
    
    pars_ac, pars_q = pars.pop('quantum')
    pars_c, pars_a  = pars_ac.pop('angles')
    
    ## Gradients
    grad_q  = jax.tree_map(jnp.zeros_like, pars_q)
    grad_a  = jax.tree_map(jnp.zeros_like, pars_a)
    grad_c  = jax.tree_map(jnp.zeros_like, pars_c)
    
    
    for (hq,hc) in h_mixed:
        σp, mels    = hc.get_conn_padded(σ_batch)
        if valid_samples:
            σp, mels    = validate_samples(hc.hilbert, σp, mels)

        if not unbiased:
            e_loc, g_a, g_q, f_c = e_loc_and_grad_batched(ma, dev, circ, pars, σ_batch, σp, mels, sample_to_angle, hq)
        else:
            e_loc, g_a, g_q, f_c = e_loc_and_grad_unbiased_batched(ma, ma_lin, dev, circ, pars, σ_batch, σp, mels, sample_to_angle, hq)


        
        # Update 
        e_locs = e_locs + e_loc
        grad_q = jax.tree_map(lambda x,y: x+jnp.mean(y, axis=0), grad_q, g_q)
        grad_a = jax.tree_map(lambda x,y: x+jnp.mean(y, axis=0), grad_a, g_a)
        grad_c = jax.tree_map(lambda x,y: x+jnp.mean(y, axis=0), grad_c, f_c)
        
        


    # Total gradient of the classical params
    e_expect    = nk.stats.statistics(e_locs.reshape(σ.shape[:2]).T)
    e_mean      = e_expect.mean.real
    Ok          = compute_Ok(ma, pars_c, σ_batch)

    grad     = jax.tree_map(lambda x,y: x-e_mean*y, grad_c, Ok)
    
    # Put the classical and quantum gradient together
    grad_all = flax.core.unfreeze(grad)
    grad_all['quantum'] = grad_q
    grad_all['angles']  = grad_a
    grad_all = flax.core.freeze(grad_all)
        
    return e_expect, grad_all
