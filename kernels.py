"""
    Kernels for the hawkes process
"""
from typing import List
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from numpy import exp

@dataclass
class State:
    """
        Captures the state (intensity) and the
        current time.
    """
    intensity: float = 0
    time: float = 0

class Kernel(metaclass=ABCMeta):
    """ abstract base class for kernels """
    @abstractmethod
    def __init__(self, *args):
        """ for setting the parameters """
        pass

    @abstractmethod
    def advancestate(self, state: State, time: float):
        """ advances the state (can use approximations) """
        pass

    @abstractmethod
    @staticmethod
    def addevent(scale: float):
        """ Adds an event to increase (or decrease) the intensity """
        pass

class ExponentialKernel:
    """
        Hold the parameter for an exponential kernel
        exp{-beta t}
    """
    def __init__(self, beta: float = 1.):
        self.beta = beta

    def advancestate(self, state: State, time: float):
        """ advances the state by decaying the exponential """
        deltat = time - state.time
        state.time = time
        state.intensity = state.intensity * exp(-deltat * self.beta)

    @staticmethod
    def addevent(state: State, scale: float = 1.):
        """ adds an event """
        state.intensity += scale

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

    def addevent(self, scale: float = 1.):
        """ Adds an event """
        for k in self._kernels:
            k.addevent(scale)
