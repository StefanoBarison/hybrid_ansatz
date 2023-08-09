## The file contains circuits and NN models to create the hybrid ansatze
from typing import Any, Union
from flax   import linen as nn
from netket import nn as nknn
from netket.utils.types import NNInitFunc

from cx import CX
import numpy as np
import jax.numpy as jnp


import pennylane as qml


############# NN


from jax.nn.initializers import normal
default_kernel_init = normal(stddev=0.01)
default_init        = nn.ones

## ==================== Classic ====================

# Classic models for the bath partition
class MeanField(nn.Module):
    
    # The most compact way to define the model is this.
    # The __call__(self, x) function should take as 
    # input a batch of states x.shape = (n_samples, L)
    # and should return a vector of n_samples log-amplitudes
    @nn.compact
    def __call__(self, x):
        
        # A tensor of variational parameters is defined by calling
        # the method `self.param` where the arguments will be:
        # - arbitrary name used to refer to this set of parameters
        # - an initializer used to provide the initial values. 
        # - The shape of the tensor
        # - The dtype of the tensor.
        lam = self.param(
            "lambda", nn.initializers.normal(), (1,), float
        )
        
        # compute the probabilities
        p = nn.log_sigmoid(lam*x)

        # sum the output
        return 0.5 * jnp.sum(p, axis=-1)


class JastrowShort(nn.Module):
    @nn.compact
    def __call__(self, x):
        
        # Define the two variational parameters J1 and J2
        j1 = self.param(
            "j1", nn.initializers.normal(), (1,), float
        )
        j2 =self.param(
            "j2", nn.initializers.normal(), (1,), float
        )

        # compute the nearest-neighbor correlations
        corr1=x*jnp.roll(x,-1,axis=-1)
        corr2=x*jnp.roll(x,-2,axis=-1)

        # sum the output
        return jnp.sum(j1*corr1+j2*corr2,axis=-1)

class RBM_Linear(nn.Module):
    r"""A restricted boltzman Machine 
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, input):
        N = input.shape[-1]
        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = jnp.prod(x, axis=-1)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (input.shape[-1],),
                self.param_dtype,
            )
            out_bias = jnp.dot(input, v_bias)
            out_RBM = x*jnp.exp(out_bias)
        else:
            out_RBM = x


        return out_RBM


## ==================== Sample to angle NN ======================

# Neural network that takes a classical congiguration as input and outputs a set of parameters for the quantum circuit
class Sample_to_angle(nn.Module):
    r""" 
    FFNN with a nonlinear activation function in between.
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    angles: Union[float, int] = 1
    """Number of angles"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, input):
        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = nn.Dense(
            name="Dense angles",
            features=int(self.angles),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(x)
       
        return x

    
## Neural network to use on particle preserving ansatz
class Sample_to_angle_particle_preserving(nn.Module):
    r"""
    FFNN with a nonlinear activation function in between.
    This NN is used to generate the angles for the particle preserving ansatz,
    therefore returns both the angles and the number of electrons to be add in the circuit.
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    
    angles: Union[float, int] = 1
    """Number of angles"""
    n_elec: Union[float, int] = 1
    """Number of total electrons in the molecule"""
    n_elec_alpha: Union[float, int] = 1
    """Number of total electrons with spin up in the molecule"""
    n_elec_beta: Union[float, int] = 1
    """Number of total electrons with spin down in the molecule"""
    n_orbitals:  Union[float, int] = 1
    """Number of total electrons with spin down in the molecule"""
    
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, input):
        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = nn.Dense(
            name="Dense angles",
            features=int(self.angles),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(x)
        
        ## Evaluate the number of alpha and beta electrons
        input_r = input.reshape(input.shape[:-1]+(2,input.shape[-1]//2))
        n_x = self.n_elec//2 - jnp.sum(jnp.asarray((input_r+1)/2), axis=-1)
        return jnp.concatenate([n_x,x], axis=-1)