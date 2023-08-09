from cx import CX
import numpy as np
import jax.numpy as jnp

import pennylane as qml




## PENNYLANE CIRCUITS
'''
Here we list two classes of circuits that can be used to define the quantum part of the hybrid ansatz.
The first one represents the circuits that will be used in the classically controlled part of the circuit.
These functions take as input the device, the parameters of the (purely quantum) circuit and the angles produced from the sample to angle function.
If another set of angles θ_η is needed, it can be passed as an additional argument. 
This adds an additional qubit to perform the Hadamard test.
The second class of circuits is the one that will be used to create the purely quantum part of the hybrid ansatz.
Here only the device and the parameters are needed as input.
For the classical controlled part
'''


def circ_classic_SU2(dev, params, θ_σ, θ_η):

    '''
    Classically controlled circuit that implements the SU(2) gates on the first and the last qubit.
    The number of angles needed is 6 (3 for each qubit).
    '''
    
    ## Obtain device info (Hadamard test or not)
    wires_list = list(dev.wire_map.keys())
    if "a" in wires_list:
        phys_qubits = len(wires_list) -1
    else:
        phys_qubits = len(wires_list)
        
        
    ## The classical circuit will be controlled only by the angles given
    p_count = 0
    
    ## θ_σ and θ_η should have the same dimensions
    if not (θ_η is None):
        assert θ_σ.shape == θ_η.shape
    
    
    for i in [0,phys_qubits-1]:
        
        qml.RY(θ_σ[p_count],wires=i)
        if not (θ_η is None):
            qml.ctrl(qml.RY, control='a')(θ_η[p_count]-θ_σ[p_count], wires=i)
        p_count += 1
        
        qml.RZ(θ_σ[p_count],wires=i)
        if not (θ_η is None):
            qml.ctrl(qml.RZ, control='a')(θ_η[p_count]-θ_σ[p_count], wires=i)
        p_count += 1
    
        qml.RY(θ_σ[p_count],wires=i)
        if not (θ_η is None):
            qml.ctrl(qml.RY, control='a')(θ_η[p_count]-θ_σ[p_count], wires=i)
        p_count += 1
  
    #=====================
    
    qml.Barrier(only_visual=True)
    


#============================================================================================================
    
def circ_classic_RY(dev, params, θ_σ, θ_η):
    
    '''
    Classically controlled circuit that implements the Ry gates on the first and the last qubit.
    The number of angles needed is 2 (1 for each qubit).
    '''
    
    ## Obtain qubit info
    wires_list = list(dev.wire_map.keys())
    if "a" in wires_list:
        phys_qubits = len(wires_list) -1
    else:
        phys_qubits = len(wires_list)
        
        
    ## The classical circuit will be controlled only by the angles given
    p_count = 0
    
    ## θ_σ and θ_η should have the same dimensions
    if not (θ_η is None):
        assert θ_σ.shape == θ_η.shape
        
    for i in [0,phys_qubits-1]:
        qml.RY(θ_σ[p_count],wires=i)
        if not (θ_η is None):
            qml.ctrl(qml.RY, control='a')(θ_η[p_count]-θ_σ[p_count], wires=i)
        p_count += 1
    #==============
    
    qml.Barrier(only_visual=True)
    


#============================================================================================================

##### NB This circuit is specific for a 2 orbitals active space

def molecule_jw_init_singles(dev,params,θ_σ,θ_η):
    '''
    This function creates the circuit with the initial occupation of the active space and adds single excitations,
    which are controlled by the angles given in θ_σ and θ_η.
    The number of angles needed is 2 (1 for each single excitation).
    '''
    
    ## Obtain qubit info
    wires_list = list(dev.wire_map.keys())
    if "a" in wires_list:
        phys_qubits = len(wires_list) -1
    else:
        phys_qubits = len(wires_list)
        
    if not (θ_η is None):
        assert θ_σ.shape == θ_η.shape
        
    ## Now we have to determine how to initialise the circuit given a string
    ## The first two angles indicate how many electrons to put in the circuit
    
    ii = jnp.arange(phys_qubits//2).reshape(1,-1)

    n_updowns = jnp.array([[θ_σ[0].astype(jnp.int64)], [θ_σ[1].astype(jnp.int64)]])
    x = (ii < n_updowns).astype(jnp.int64).reshape(-1)
    
    if (θ_η is None):
        for i in range(phys_qubits):
            CX(1-x[i],wires=i) ## CX(0) == X, CX(1) == I
    else:
        ## Anti-controlled x
        for i in range(phys_qubits):
            qml.ctrl(CX, control='a',control_values=0)(1-x[i],wires=i)
        
        ## Controlled x'
        np_updowns = jnp.array([[θ_η[0].astype(jnp.int64)], [θ_η[1].astype(jnp.int64)]])
        xp = (ii < np_updowns).astype(jnp.int64).reshape(-1)
        for i in range(phys_qubits):
            qml.ctrl(CX, control='a')(1-xp[i],wires=i)
        
    qml.Barrier()
    ## Now we put the controlled rotations
    
    qml.SingleExcitation(θ_σ[2], wires=[0, 1])
    qml.SingleExcitation(θ_σ[3], wires=[2, 3])
    
    if not (θ_η is None):
        qml.ctrl(qml.SingleExcitation, control='a')(θ_η[2]-θ_σ[2],wires=[0, 1])
        qml.ctrl(qml.SingleExcitation, control='a')(θ_η[3]-θ_σ[3],wires=[2, 3])
     
    qml.Barrier()
    
    

#============================================================================================================
# Purely quantum circuits

def circ_pass(*args):
    '''This function does nothing, it is used to pass a function to the circuit without doing anything'''
    pass

            

def circ_RY(depth, dev,  params):
    '''
    Hardware efficient ansatz consisting of layers of single qubit Ry rotations and CNOTs.
    The number of layers is given by depth.
    The total number of parameters is (depth+1)*number of qubits.
    '''
    
    ## Obtain qubit info
    wires_list = list(dev.wire_map.keys())
    if "a" in wires_list:
        phys_qubits = len(wires_list) -1
    else:
        phys_qubits = len(wires_list)
    
    count = 0 
    
    for _ in range(depth):
        for i in range(phys_qubits):
            qml.RY(params[count],wires=i)
            count += 1
    
        for i in range(phys_qubits-1):
            qml.CNOT(wires=[i,i+1])
            
    # Final layer of single qubits rotations
    for i in range(phys_qubits):
        qml.RY(params[count],wires=i)
        count += 1
        




## Circuits for the chemical HONO-LUNO active space
def circ_chemistry_double(dev,pars):
    '''
    This function adds the double excitation to the circuit. It is specific for the HONO-LUNO active space.
    It needs 1 parameter.
    '''
    
    wires_list = list(dev.wire_map.keys())
    if "a" in wires_list:
        phys_qubits = len(wires_list) -1
    else:
        phys_qubits = len(wires_list)
    
    qml.DoubleExcitation(pars[0], wires=[0, 2, 1, 3]) ## Right excitation 1010 <-> 0101
    
   
    qml.Barrier()
