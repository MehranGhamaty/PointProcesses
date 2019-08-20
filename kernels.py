"""
    Kernels for the hawkes process
"""
from typing import List
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import tensorflow as tf

@dataclass
class State:
    """
        Captures the state (intensity) and the
        current time.
    """
    intensity: tf.float32 = 0.
    time: tf.float32 = 0.

class Kernel(metaclass=ABCMeta):
    """ abstract base class for kernels """
    @abstractmethod
    def __init__(self, *args):
        """ for setting the parameters """
        pass

    @abstractmethod
    def advancestate(self, state: State, time: tf.float32):
        """ advances the state (can use approximations) """
        pass

    @staticmethod
    @abstractmethod
    def addevent(state: State, scale: tf.float32):
        """ Adds an event to increase (or decrease) the intensity """
        pass

class ExponentialKernel:
    """
        Hold the parameter for an exponential kernel
        exp{-beta t}
    """
    def __init__(self, beta: tf.float32 = 1.):
        self.beta = beta

    def advancestate(self, state: State, time: tf.float32) -> State:
        """ advances the state by decaying the exponential """
        deltat = time - state.time
        state.time = time
        state.intensity = state.intensity * tf.math.exp(-deltat * self.beta)
        return state

    @staticmethod
    def addevent(state: State, scale: tf.float32 = 1.) -> State:
        """ adds an event """
        state.intensity += scale
        return state

class SumExponentialKernel:
    """
        Sum of exponentials
    """
    def __init__(self, betas: List[float]):
        self._kernels = []
        for beta in betas:
            self._kernels.append(ExponentialKernel(beta))

    def advancestate(self, state: State, time: float):
        """ advances all of the states """
        for k in self._kernels:
            k.advancestate(state, time)

    @staticmethod
    def addevent(state: State, scale: tf.float16 = 1.):
        """ Adds an event """
        state.intensity += scale
