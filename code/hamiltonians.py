import numpy as np

import pennylane as qml
import netket as nk
from netket.operator.spin import sigmax,sigmaz, sigmay

from functools import partial, reduce
import pickle
import json

################################################################################################
################################################################################################

## Function to generate a list of Pauli strings for the Ising model

def generate_ising_hami_list(n_spins,J_list,h_list):
    """
    Generate the list of Pauli strings for the Ising model.
    Args:
        spins: number of spins
        J_list: list of J coefficients
        h_list: list of h coefficients
    """
    op_list = []
    id = "I"*n_spins

    for i in range(n_spins-1):
        # Append the sigma Z term
        op = list(id)
        op[i] = "Z"
        op[i+1] = "Z"
        op = "".join(op)
        op_list.append(op)

    for i in range(n_spins):
        # Append the sigma X term
        op = list(id)
        op[i] = "X"
        op = "".join(op)
        op_list.append(op)

    

    return op_list, J_list + h_list

################################################################################################


## Hamiltonians from lists of Pauli strings

def pennylane_hami_from_list(pauli_list, coeffs, wires):

    '''
    This function takes a list of Pauli strings and of coefficients and returns a Pennylane Hamiltonian
    '''

    paulis = []
    for op in pauli_list:
        paulis.append(qml.pauli.string_to_pauli_word(op))

    p_op = qml.Hamiltonian(coeffs,paulis)

    return p_op


def netket_hami_from_list(pauli_list, coeffs, hi):
    '''
    This function takes a list of Pauli strings, of coefficients and the Netket Hilbert space and returns a Netket Hamiltonian
    '''
    nk_op = nk.operator.PauliStrings(hi, pauli_list,coeffs)
   
    return nk_op


def mixed_hami_from_list(pauli_list, coeffs, hi, q_index, c_index):
    '''
    This function takes a list of Pauli strings, of coefficients and the Netket Hilbert space and returns a mixed Hamiltonian 
    Args:
        pauli_list: list of Pauli strings
        coeffs: list of coefficients
        hi: Netket Hilbert space
        q_index: indexes of the quantum partition
        c_index: indexes of the classical partition
    '''

    tot_index  = list(range(len(pauli_list[0])))
        
    # Now separe the lists and create the mixed operator
    q_op_list    = []
    c_op_list    = []
    m_op_list    = []
    
    mc_op_list   = []
    mq_op_list   = []
    
    q_coeff_list = []
    c_coeff_list = []
    m_coeff_list = []
    
    for (op, coeff) in zip(pauli_list, coeffs):
        c_term = False
        q_term = False
        
        for i in tot_index:
            
            # Check that it has a quantum term
            if op[i] != 'I' and  (i in q_index):
                q_term = True
                
            # Check that it has a classical term
            if op[i] != 'I' and  (i in c_index):
                c_term = True
                
        # Now put in the right list
        
        if not c_term and not q_term:
            # This is the identity
            q_op_list.append("".join([op[k] for k in q_index]))
            q_coeff_list.append(coeff.real)
            
        if c_term and not q_term:
            c_op_list.append("".join([op[k] for k in c_index]))
            c_coeff_list.append(coeff.real)
            
        if q_term and not c_term:
            q_op_list.append("".join([op[k] for k in q_index]))
            q_coeff_list.append(coeff.real)
            
        if c_term and q_term:
            #print(coeff,op)
            m_op_list.append("".join([op[k] for k in tot_index]))
            mc_op_list.append("".join([op[k] for k in c_index]))
            mq_op_list.append("".join([op[k] for k in q_index]))
            
            m_coeff_list.append(coeff.real)
    
    # Determine the different classical string in mixed operators
    mc_op_set = set(mc_op_list)
    mq_op_set = set(mq_op_list)
    
    print("Number of pure  quantum op:",len(q_op_list))
    print("Number of pure  classic op:",len(c_op_list))
    print("Number of       mixed   op:",len(m_op_list))
    print("Number of quantum    group:",len(mq_op_set))
    
    
    # Now that we have the three lists we can create Pennylane and netket operators
    #hi       = nk.hilbert.Spin(s=1 / 2, N=len(c_index))
    mixed_op = []
    
    # 1) Quantum op
    hq = partial(pennylane_hami_from_list,q_op_list,q_coeff_list)
    #hq = lambda wires : pennylane_hami_from_list(q_op_list, q_coeff_list, wires)
    # Classic identity
    idc = nk.operator.LocalOperator(hi, constant=1)
    
    mixed_op.append((hq,idc))
    
    # 2) Classic op
    hc  = nk.operator.PauliStrings(hi, c_op_list,c_coeff_list)
    # Quantum identity
    idq = lambda wires: qml.Identity(wires[0])
    
    mixed_op.append((idq,hc))
    
    
    # 3) Mixed op
    ## Grouping quantum operators
    
    for q_op in mq_op_set:
        c_op     = []
        c_coeffs = []
        for (i,op) in enumerate(mq_op_list):
            if op == q_op:
                c_op.append(mc_op_list[i])
                c_coeffs.append(m_coeff_list[i])
        
        hq_i  = partial(pennylane_hami_from_list, [q_op], [1.0])
        hc_i  = nk.operator.PauliStrings(hi, c_op, c_coeffs)
        mixed_op.append((hq_i,hc_i))

    
    return mixed_op


################################################################################################
## Mixed Hamiltonian from files that contains paulis strings saved in 
## dictionaries with the coefficients, represented as tuples of non identity pau as keys --> {((pos_0,pauli_0),(pos_1,pauli_1),...,(pos_k,pauli_k)):coeff}

def print_hami_from_file(filename, tot_spins):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # Prepare the operator list
    pauli_list = []
    coeffs     = []
    
    for key in data.keys():
    
        pauli_string = list(tot_spins*"I")
        for term in key:
            pauli_string[term[0]] = term[1]
        
        pauli_string = ''.join(pauli_string)
        pauli_list.append(pauli_string)
        coeffs.append(data[key].real)

    return pauli_list, coeffs

def netket_hami_from_file(filename, hi, tot_spins):
    ## Load openfermion file
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    # Prepare the operator list
    pauli_list = []
    coeffs     = []
    
    for key in data.keys():
    
        pauli_string = list(tot_spins*"I")
        for term in key:
            pauli_string[term[0]] = term[1]
        
        pauli_string = ''.join(pauli_string)
        pauli_list.append(pauli_string)
        coeffs.append(data[key].real)
    
        
    ## Now create the Netket operator with PauliStrings
    
    #hi    = nk.hilbert.Spin(s=1 / 2, N=len(pauli_list[0]))
    nk_op = nk.operator.PauliStrings(hi, pauli_list,coeffs)
    
    return nk_op


def mixed_hami_from_file(filename, hi, tot_spins, q_index, c_index):

    '''
    This function creates a mixed hamiltonian from a file that contains pauli strings saved in
    dictionaries with the coefficients, represented as tuples of non identity paulis as keys --> {((pos_0,pauli_0),(pos_1,pauli_1),...,(pos_k,pauli_k)):coeff}
    Args:
        filename: name of the file
        hi: Hilbert space of the classic part
        tot_spins: total number of spins
        q_index: list of the quantum spins
        c_index: list of the classic spins
    '''
    ## Load openfermion file
    
    with open(filename, "rb") as f:
        data = pickle.load(f)
        
    # Prepare the operator list
    pauli_list = []
    coeffs     = []
    tot_op     = []
    
    for key in data.keys():
    
        pauli_string = list(tot_spins*"I")
        for term in key:
            pauli_string[term[0]] = term[1]
        
        pauli_string = ''.join(pauli_string)
        pauli_list.append(pauli_string)
        coeffs.append(data[key].real)
        tot_op.append((pauli_string,data[key].real))
        
    tot_index  = list(range(len(pauli_list[0])))
        
    # Now separe the lists and create the mixed operator
    q_op_list    = []
    c_op_list    = []
    m_op_list    = []
    
    mc_op_list   = []
    mq_op_list   = []
    
    q_coeff_list = []
    c_coeff_list = []
    m_coeff_list = []
    
    for (op, coeff) in tot_op:
        c_term = False
        q_term = False
        
        for i in tot_index:
            
            # Check that it has a quantum term
            if op[i] != 'I' and  (i in q_index):
                q_term = True
                
            # Check that it has a classical term
            if op[i] != 'I' and  (i in c_index):
                c_term = True
                
        # Now put in the right list
        
        if not c_term and not q_term:
            # This is the identity
            q_op_list.append("".join([op[k] for k in q_index]))
            q_coeff_list.append(coeff.real)
            
        if c_term and not q_term:
            c_op_list.append("".join([op[k] for k in c_index]))
            c_coeff_list.append(coeff.real)
            
        if q_term and not c_term:
            q_op_list.append("".join([op[k] for k in q_index]))
            q_coeff_list.append(coeff.real)
            
        if c_term and q_term:
            #print(coeff,op)
            m_op_list.append("".join([op[k] for k in tot_index]))
            mc_op_list.append("".join([op[k] for k in c_index]))
            mq_op_list.append("".join([op[k] for k in q_index]))
            
            m_coeff_list.append(coeff.real)
    
    # Determine the different classical string in mixed operators
    mc_op_set = set(mc_op_list)
    mq_op_set = set(mq_op_list)
    
    print("Number of pure  quantum op:",len(q_op_list))
    print("Number of pure  classic op:",len(c_op_list))
    print("Number of       mixed   op:",len(m_op_list))
    print("Number of quantum    group:",len(mq_op_set))
    
    
    # Now that we have the three lists we can create Pennylane and netket operators
    #hi       = nk.hilbert.Spin(s=1 / 2, N=len(c_index))
    mixed_op = []
    
    # 1) Quantum op
    hq = partial(pennylane_hami_from_list,q_op_list,q_coeff_list)
    #hq = lambda wires : pennylane_hami_from_list(q_op_list, q_coeff_list, wires)
    # Classic identity
    idc = nk.operator.LocalOperator(hi, constant=1)
    
    mixed_op.append((hq,idc))
    
    # 2) Classic op
    hc  = nk.operator.PauliStrings(hi, c_op_list,c_coeff_list)
    # Quantum identity
    idq = lambda wires: qml.Identity(wires[0])
    
    mixed_op.append((idq,hc))
    
    
    # 3) Mixed op
    ## Grouping quantum operators
    
    for q_op in mq_op_set:
        c_op     = []
        c_coeffs = []
        for (i,op) in enumerate(mq_op_list):
            if op == q_op:
                c_op.append(mc_op_list[i])
                c_coeffs.append(m_coeff_list[i])
        
        hq_i  = partial(pennylane_hami_from_list, [q_op], [1.0])
        hc_i  = nk.operator.PauliStrings(hi, c_op, c_coeffs)
        mixed_op.append((hq_i,hc_i))

    
    return mixed_op
