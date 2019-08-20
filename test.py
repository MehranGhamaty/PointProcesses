
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from trajectory import Trajectory
from kernels import Kernel, State

from hawkes import Hawkes

    def __init__(self, mus: tf.Tensor, W: tf.Tensor, kernel: Kernel, states: List[State]):
        self._mus: tf.Tensor = mus
        self._weights: tf.Tensor = W
        self._kernel: Kernel = kernel
        self._states: List[State] = states


hp = Hawkes(mus, weights, kernel, states)
