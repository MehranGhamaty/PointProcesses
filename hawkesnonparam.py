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
    def __init__(self, nlabels: int, kernel: Callable[[float], float], threshold: float ):
        self.__nlabels = nlabels
        self.__mus = np.random.random(nlabels)
        self.__functions = np.array([[kernel] for i in range(nlabels)], dtype=object)
        self.__threshold = threshold #only care about events within this value
        self.__internalqueue = list() #this hold the timeslices that are within the period

    
    def lambda(event: TimeSlice): 
            """
                Returns the current values for lambda 

            """
            state = np.array([0 for i in range(nlabels)])

            #maintain our queue
            while self.__internalqueue[0].time < event.time - z: 
                del self.__internalqueue[0]

            self.__internalqueue.push(event)

            #now update our state
            for ev in self.__internalqueue:
                print(ev)
                state

            return state + self.__mus


    def __repr__(self):
        return self.__mus.__repr__()
