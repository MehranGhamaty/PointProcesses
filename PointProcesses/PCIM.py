
"""
    Author: Mehran Ghamaty

    This implementation of a PCIM that is restricted to
    change points that occur specifially on events.
    This is done as to use the scikitlearn package.

    Other implementations have access to the entire trajectory which allows
    for picking the counts all the time

    Giving up on this for a little while, I think Hawkes process might be a bit
    better. I guess the problem is either implmenting this or implementing EM for the 
    Hawkes process. Both are fairly challenging but possible with TF.

"""
from abc import ABCMeta, abstractmethod
from typing import Optional, List, Tuple, Dict

import tensorflow as tf
import numpy as np

from PointProcess import PointProcess
from PoissonProcess import PoissonProcess

from DataStores.Trajectory import Trajectory, TimeSlice

class BasisFunction(metaclass=ABCMeta):
    """
        ABC for basis functions.

        These all have an internal state (for online learning its better to
        keep track of everything).
    """
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, time_slice: TimeSlice) -> bool:
        """
            Takes a list of (ordered) time slices
        """

    @abstractmethod
    def eval(self) -> bool:
        """ gets the current state """

    @abstractmethod
    def __repr__(self):
        """ string representation """

    @abstractmethod
    def resetstate(self):
        """ Resets internal state """

class CountBasisFunction(BasisFunction):
    """
        Splits if there have been more events of a specific type in the past
        [t-lag1, t-lag0] window.

        -------------------->
        t-lag1    t-lag0    t

        if label == -1 then its if any type
    """
    def __init__(self, label, lag0, lag1, n):
        assert lag0 > lag1
        super(CountBasisFunction, self).__init__()
        self.__labelofinterest = label
        self.__lag0 = lag0
        self.__lag1 = lag1
        self.__n = n
        self.__currnumber = 0
        self.__queue = []

    def __call__(self, time_slice: TimeSlice) -> bool:
        """
            Adds the time slice
        """
        self.__queue.append(time_slice)

        while self.__queue[0].time < self.__queue[-1].time - self.__lag1:
            del self.__queue[0]

        self.__currnumber = len(self.__queue)

        while self.__queue[self.__currnumber-1] - time_slice.time < self.__lag0:
            self.__currnumber -= 1

        return self.eval()

    def eval(self):
        """
            gets the current state
        """
        return self.__currnumber > self.__n

    def resetstate(self):
        self.__queue = []
        self.__currnumber = 0

    def __repr__(self):
        return "More than {} events of type {} in the past [{}, {}]?".format(
            self.__n, self.__labelofinterest, self.__lag1, self.__lag0)

class BasisFunctionBank:
    """
        Manages tests
        and the scores associated with
        each basis function so far.



        Takes a variable and counts

    """
    def __init__(self, variables_and_counts: Dict[int, List[int]],
                 time_windows: List[Tuple[float, float]]):
        self.__variables_and_counts = variables_and_counts
        self.__time_windows = time_windows
        self.__basis_functions: List[BasisFunction] = \
            self._generate_basis_function(self.__variables_and_counts, self.__time_windows)
        self.__segments: List[Tuple[List[TimeSlice], List[TimeSlice]]] = \
            [(list(), list()) for _ in len(self.__basis_functions)]
        self.__counts: List[Tuple[int, int]] = [(0,0) for _ in len(self.__basis_functions)]
        self.__timeamounts: List[Tuple[float, float]] = [(0., 0.) for _ in len(self.__basis_functions)]
        self.__scores: List[Tuple[float, float]] = [(0., 0.) for _ in len(self.__basis_functions)]
        self.__i: int = 0
        self.__highest: int = -1
        self.__secondhighest: int = -1

    def _generate_basis_function(self, variables_and_counts: Dict[int, List[int]],
                                 time_windows: List[Tuple[float, float]]) -> List[BasisFunction]:
        """
            Generates basis functions
            This entire design is wrong.
        """
        return []

    @property
    def highest(self) -> BasisFunction:
        """ returns the higest scoring basis function """
        return self.__basis_functions[self.__highest]


    def resetstate(self):
        """
            Resets the internal state of the testbank
        """
        self.__segments = [(list(), list()) for _ in len(self.__basis_functions)]
        self.__counts = [(0, 0) for _ in len(self.__basis_functions)]
        self.__timeamounts = [(0., 0.) for _ in len(self.__basis_functions)]
        self.__scores = [(0., 0.) for _ in len(self.__basis_functions)]
        self.__i = 0
        self.__highest = -1
        self.__secondhighest = -1

    def add_time_slice(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            adds a time slice to each bank and returns a tensor
            that contains the score per item that changed
        """
        for i, basis_fun in enumerate(self.__basis_functions):
            """
                So for each item I call it and it returns true and false
                That is for each basis function I need to keep track
                the time_slices that go into the true and false
                branches. I also need to keep track of each scores
            """
            evaluation = basis_fun(time_slice)
            self.__segments[i][evaluation].append(time_slice)
            self.__timeamounts[i][evaluation] += time_slice.deltat
            if time_slice.label != -1:
                self.__counts[i][evaluation] += 1

            """
            I would need to recalc all the true and false branches
             to get the llh and thats what checks if its done.
             the llh is then
             n * log(n/t) - t * n/t
             n * log(n/t) - n
             (this is for just llh....)

             for each side

             So for the basis functions I can't be using a static list....

             Maybe something else.....

             I could try to just keep the sufficient statistics about the processes.

             What would the goal then be?

             so everytime I add an event what exactly do I do?

             I need to have a queue per time window.
             Is there any way to

             What have I done since monday (not including all the stuff on the desktop)
             I have fixed the hawkes process stuff, I ran the non-conjoint tests.

             I also don't have any of the Bayes stuff, which seemed to not make a real difference
             other than the selection of kappa or whatever.
            """
            n = self.__counts[i][evaluation]
            t = self.__timeamounts[i][evaluation]
            self.__scores[i][evaluation] =  n * np.log(n/t) - n

            totalscore = self.__scores[i][0] + self.__scores[i][1]
            highscore = self.__scores[self.__highest][0] + self.__scores[self.__highest][1]
            if totalscore > highscore:
                self.__highest = i
            elif (totalscore >
                  self.__scores[self.__secondhighest][0] + self.__scores[self.__secondhighest][1]):
                self.__secondhighest = i

        scores = [p[0] + p[1] for p in self.__scores]
        return tf.convert_to_tensor(scores, tf.Tensor)

    def checkdone(self, delta) -> bool:
        """
            Checks to see if the top scoring item is
            far enough ahead of the second top scoring item
        """
        return self.__highest > self.__secondhighest + delta

    def __iter__(self):
        """ to iterate through all the basis functions """
        return self

    def __next__(self) -> BasisFunction:
        if self.__i >= len(self.__basis_functions):
            self.__i = 0
            raise StopIteration
        self.__i += 1
        return self.__basis_functions[self.__i]


class DTNode:
    """
        This is the DT Node which contains some PointProcess at its leaves.

    """
    def __init__(self, variables_and_counts: Dict[int, List[int]],
                 time_windows: List[Tuple[float, float]],
                 delta: float = 10.):
        self.__variables_and_counts = variables_and_counts
        self.__time_windows = time_windows
        self.__delta = delta
        self.__bank: BasisFunctionBank = BasisFunctionBank(self.__variables_and_counts, self.__time_windows)
        self.__time_slices: List[TimeSlice] = list()
        self.__basis_function: Optional[BasisFunction] = None
        self.__distribution: PointProcess = PoissonProcess(len(self.__variables))
        self.__done: bool = False
        self.__branches: Optional[List[DTNode]] = None

    def resetstate(self):
        """
            Resets the internal state
        """
        self.__bank = BasisFunctionBank(self.__variables_and_counts, self.__time_windows)
        self.__time_slices = list()
        self.__basis_function = None
        self.__distribution = PoissonProcess(len(self.__variables))
        self.__done = False
        self.__branches = None

    def add_time_slice(self, time_slice: TimeSlice) -> tf.Tensor:
        """
            add time slice to each test and update all the scores

            returns the intensities from the processes at leaves
        """
        if self.__done:
            """
                I need to check to see where this time_slice
                goes
            """
            evaluation = self.__basis_function(time_slice)
            self.__branches[evaluation].add_time_slice(time_slice)
        else:
            self.__bank.add_time_slice(time_slice)
            self.check_if_done()

    def check_if_done(self):
        """
            checks to see if the highest score has a far enough lead
        """
        if self.__bank.check_if_done(self.__delta):
            self.__done = True
            self.__basis_function = self.__bank.highest
            self.__branches = [DT(self.__nl)]

    def freeze(self):
        """
            sets done to true so that the parameters aren't updated continuously
        """
        self.__done = True

class DT(PointProcess):
    """
        This class manges a structure of PCIMNodes.

        It keeps track of the root node, the total times,
        along with all the leaves.
    """
    def __init__(self, nlabels: List[int], test_bank: BasisFunctionBank):
        super(DT, self).__init__()
        self.__testbank = testbank
        self.__nlabels = nlabels
        self.__root: PCIMNode = PCIMNode()
        self.__total_time = 0
        self.__leaves: [self.__root]

    def resetstate(self):
        """
            Resets internal state of self and all the leaf nodes
        """
        for leaf in self.__leaves:
           leaf.resetstate()
        self.__total_time = 0

    def calcllh(self, traj: Trajectory) -> tf.Variable:
        """
            calculates the log likelihood over the entire
            trajectory.
        """

    def _get_rate(self, variable) -> float:
        """
            gets the rate for a variable.
        """
        return self.__root.get_rate()

    def __call__(self, time_slice: TimeSlice) -> tf.Variable:
        """
            Calls to add a time_slice to the state and returns the intensity per
            variable.
        """
        return tf.convert_to_tensor([self._get_rate(v) for v in range(self.__nlabels)], dtype=tf.dtype.float32)
