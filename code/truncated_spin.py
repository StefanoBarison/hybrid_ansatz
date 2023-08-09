# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fractions import Fraction
from typing import Optional, List, Union
from functools import partial
from math import factorial

import numpy as np
from netket.graph import AbstractGraph
from numba import jit

import jax
import jax.numpy as jnp
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.hilbert.random import random_state
from netket.hilbert import TensorHilbert


@jit(nopython=True)
def _sum_constraint(x, min_up, max_up):
    xs = np.sum(((x + 1) / 2).astype(np.int32), axis=1)
    return np.logical_and(xs <= max_up, xs >= min_up)


class Spin(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local spin states."""

    def __init__(
        self,
        N: int = 1,
        *,
        minimum_up: Optional[float] = None,
        maximum_up: Optional[float] = None,
    ):
        r"""Hilbert space obtained as tensor product of local spin states.

        Args:
           s: Spin at each site. Must be integer or half-integer.
           N: Number of sites (default=1)
           maximum_up: If given, constrains the maximum number of spin up of system to a particular value.

        Examples:
           Simple spin hilbert space.

           >>> import netket as nk
           >>> hi = nk.hilbert.Spin(s=1/2, N=4)
           >>> print(hi.size)
           4
        """
        s = 1 / 2
        local_size = 2
        local_states = np.empty(local_size)

        for i in range(local_size):
            local_states[i] = -round(2 * s) + 2 * i
        local_states = local_states.tolist()

        if maximum_up is not None:
            if maximum_up > N or maximum_up < 0:
                raise ValueError(
                    f"{maximum_up=} should be smaller than the number of spins {N=} "
                )
        if minimum_up is not None:
            if minimum_up > N or minimum_up < 0:
                raise ValueError(
                    f"{minimum_up=} should be smaller than the number of spins {N=} and larger than 0"
                )

        if maximum_up is not None or minimum_up is not None:
            if minimum_up is None:
                minimum_up = 0
            if maximum_up is None:
                maximum_up = N

            constraints = partial(_sum_constraint, min_up=minimum_up, max_up=maximum_up)

        self._minimum_up = minimum_up
        self._maximum_up = maximum_up
        self._s = s

        super().__init__(local_states, N, constraints)

    @property
    def maximum_up(self):
        return self._maximum_up

    @property
    def minimum_up(self):
        return self._minimum_up

    def __pow__(self, n):
        return TensorHilbert(*tuple(self for _ in range(n)))

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        if self._s == other._s:
            if self.maximum_up is None and other.maximum_up is None and self.minimum_up is None and other.minimum_up is None:
                return Spin(s=self._s, N=self.size + other.size)

        return NotImplemented

    def states_to_local_indices(self, x):
        numbers = (x + self.local_size - 1) / 2
        return numbers.astype(np.int32)

    def __repr__(self):
        total_sz = (
            f", maximum_up={self.maximum_up}" if self.maximum_up is not None else ""
        )
        total_sz2 = (
            f", minimum_up={self.minimum_up}" if self.minimum_up is not None else ""
        )
        return f"TruncatedSpin(s=1/2, N={self.size}{total_sz2}{total_sz})"

    @property
    def _attrs(self):
        return (self.size, self.maximum_up, self.minimum_up)


@random_state.dispatch
def random_state_truncated_qubit(hilb: Spin, key, batches: int, *, dtype):
    if hilb._maximum_up is None:
        rs = jax.random.randint(key, shape=(batches, hilb.size), minval=0, maxval=2)
        return jnp.asarray(rs, dtype=dtype)
    else:
        k1, k2 = jax.random.split(key, 2)

        # compute possible number of ups
        N_ups = range(hilb.minimum_up, hilb.maximum_up + 1)
        # Probability of each number of ups
        n_fac = factorial(hilb.size)
        configs_N_ups = [
            n_fac / (factorial(n) * factorial(hilb.size - n)) for n in N_ups
        ]
        prob_N_ups = jnp.array(np.array(configs_N_ups) / np.sum(configs_N_ups))

        # Select a certain value of N_ups
        n_chosen = jax.random.choice(
            k1, jnp.array(N_ups), p=prob_N_ups, shape=(batches,)
        ).reshape(-1, 1)

        # Generate a canonical state with that number of ups
        ii = jnp.arange(hilb.size).reshape(1, -1)
        canonical_states = (ii < n_chosen).astype(dtype) * 2 - 1

        return jax.random.permutation(k2, canonical_states, axis=-1, independent=True)
