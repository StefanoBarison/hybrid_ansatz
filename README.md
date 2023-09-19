# Embedding quantum circuit in classical variational methods

This repository contains the code to reproduce the results of the paper ["Embedding Classical Variational Methods in Quantum Circuits"](https://arxiv.org/abs/2309.08666).

## Download the code

You can download all the code present in this repository by going to the directory in which you want to save it and execute

```
git clone https://github.com/StefanoBarison/hybrid_ansatz.git
```

## Libraries installation

The simulations are performed in Python using [Netket](https://www.netket.org/) library to create the classical model (might it be a simple mean field ansatz or a Neural Quantum State) and [Pennylane](https://pennylane.ai/) for the quantum circuits.
Once you are in the downloaded directory you can install all the required packages by executing

```
pip install -r requirements.txt
```

## Repository content

- **code**: contains the code to perform quantum-classical embedded calculations
   
    - `hamiltonian.py`: functions to create the Hamiltonians in Netket and Pennylane, loading terms from a file or from a list
    - `quantum_circuits.py`: variational circuits used in the experiments, see the manuscript for more details
    - `classic_models.py`: code for the classic models used, from the mean field approximations to RBMs
    - `utils.py`: contains all the functions to compute the quantum overlaps and the Monte-Carlo expectation values, see the manuscript for more details
    - `truncated_spin.py`: a custom Netket Hilbert space that samples configurations only in a given range of particles
    - `cx.py`: a custom Pennylane gate to initialise the particle-preserving quantum circuit and perform Hadamard tests if required

   Moreover, two Jupyter notebooks are included, containing example calculations on the Ising model and on the ammonia molecule.

- **data**: contains the results of the calculations on the Ising model and on the ammonia molecule, used to produce the plot in the paper
- **plots**: the plots presented in the main text of the paper and in the supplementary material