"""
    Non parameteric version of the hawkes process

    Not sure if I can use the tensorflow stuff because I need a functional matrix 
"""

from dataclasses import dataclass
import numpy as np
from typing import Callable


class NonParametericHawkes:
    """
        For use with online learning
    """

    #pretty sure I need a factory function or something
    def __init__(self, nlabels: int, kernel: Callable[[float], float] ):
        self.__nlabels = nlabels
        self.__mus = np.random.random(nlabels)
        self.__functions = np.array([[kernel] for i in range(nlabels)], dtype=object)

    def lambda(i: int, 

    def __repr__(self):
        return self.__mus.__repr__()
