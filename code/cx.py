"""
This submodule contains the discrete-variable quantum operations that are the
core parameterized gates.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access,invalid-overridden-method
import functools
import math
from operator import matmul

import numpy as np

import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.non_parametric_ops import Hadamard, PauliX, PauliY, PauliZ
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires

INV_SQRT2 = 1 / math.sqrt(2)
stack_last = functools.partial(qml.math.stack, axis=-1)


def _can_replace(x, y):
    """
    Convenience function that returns true if x is close to y and if
    x does not require grad
    """
    return (not qml.math.requires_grad(x)) and qml.math.allclose(x, y)

class CX(Operation):
    r"""
    Class for a gate that put an X on the wire i if it receives a 0 as input, otherwise it does nothing.
    Used to combine with JAX the initialisation of the circuit.
    """
    num_wires = 1
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    basis = "X"
    grad_method = "A"
    parameter_frequencies = [(1,)]

    

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(phi, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_matrix(theta):  # pylint: disable=arguments-differ
        
        I = qml.Identity.compute_matrix()
        Rx = qml.PauliX.compute_matrix()

        return qml.math.where(theta, qml.Identity.compute_matrix(), qml.PauliX.compute_matrix())

        # The following avoids casting an imaginary quantity to reals when backpropagating
        #return qml.math.stack([stack_last([c, js]), stack_last([js, c])], axis=-2)

    def adjoint(self):
        return CX(self.data[0], wires=self.wires)

    def pow(self, z):
        return [CX(self.data[0] * z, wires=self.wires)]

    def _controlled(self, wire):
        raise NotImplementedError()
        #new_op = CRX(*self.parameters, wires=wire + self.wires)
        #return new_op.inv() if self.inverse else new_op

    def simplify(self):
        if _can_replace(self.data[0] , 0):
            return qml.Identity(wires=self.wires)

        return CX(self.data[0], wires=self.wires)
