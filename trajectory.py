"""
    A class to hold trajectories of events that contain labels.
    used for online streaming. Contains classes for a TimeSlice,
    Field, Trajectory, and Episodes.


    A TimeSlince  contains the real time stamp, the time since the last event,
    a label and a mark.

    A Field is one of the members in a Trajectory.

    A Trajectory manages a set of Fields.

    An Episodes object manages a set of Trajectories.
"""
import functools

from typing import Dict, Union, List, Set, Tuple
from dataclasses import dataclass, field
from numpy import inf

import tensorflow as tf


@dataclass
class TimeSlice:
    """
        Each of these serves as an example for the dataset.
    """
    time: tf.constant  # the end time of the slice
    deltat: tf.constant  # the duration of the slice
    label: tf.constant  # if -1 no event occured


@dataclass
class Field:
    """
        These hold the value for the field and the range
    """
    values: List[Union[float, int]] = field(default_factory=list)
    continuous: bool = True
    space: Union[Set[int], Tuple[float, float]] = (0., 0.)

    def __len__(self):
        return len(self.values)


class Trajectory:
    """
        Holds information about the events such as the their label and time
        This is for online learning, so when iterating through there is a
        delta such that
    """

    def __init__(self, fields: Dict[str, Field], tau: float = inf):
        """
            Here we will note that the times are now going to be paritioned
            such
            that t_0 = 0, and the following will be the next time or t_{k-1} +
            \tau,
            which ever comes first. If t_{k-1} + \tau is first then the associated
            label is -1

            The issue being when I'm going to calculate the sums
            I'm going to do a sum over the labels

            :param fields: A dictionary containing the fields
            :param delta: the discretization period.
        """

        # tau = tf.convert_to_tensor(tau, dtype=tf.float32)
        # neglab = tf.convert_to_tensor(-1, dtype=tf.int32)

        self.__numevents = dict()
        self.__totaltime = fields["times"].space[1] - fields["times"].space[0]

        self.__fields = dict()
        for key, val in fields.items():
            self.__fields[key] = Field(continuous=val.continuous,
                                       space=val.space)

        currtime = self.__fields["times"].space[0]
        for time, label in zip(fields["times"].values,
                               fields["labels"].values):
            timeuntilnextevent = time - currtime
            while timeuntilnextevent > tau:
                timeuntilnextevent -= tau
                self._addevent((tau, -1))

            if label not in self.numevents:
                self.numevents[label] = 0
            self.numevents[label] += 1

            self._addevent((timeuntilnextevent.numpy(), label.numpy()))

        self.__fields["times"].values = tf.convert_to_tensor(
            self.__fields["times"].values,
            dtype=tf.float32)
        self.__fields["labels"].values = tf.convert_to_tensor(
            self.__fields["labels"].values,
            dtype=tf.int32)
        self._totaltime = 0
        self._i = 0

    @property
    def totaltime(self):
        """ returns the total amount of time in the traj """
        return self.__totaltime

    @property
    def numevents(self):
        """ returns a dictionary that maps the labs to number of events """
        return self.__numevents

    def addTimeSlice(self, time_slice: TimeSlice):
        """ No validation checks """
        self.__fields["times"].value.append(time_slice.deltat)
        self.__fields["labels"].value.append(time_slice.labels)

    def _addevent(self, event: Tuple[float, int]):
        """ Adds an event to the trajectory """
        self.__fields["times"].values.append(event[0])
        self.__fields["labels"].values.append(event[1])

    @property
    def field(self) -> Dict[str, Field]:
        """ Returns the fields """
        return self.__fields

    def __len__(self) -> int:
        return len(self.__fields["times"])

    def __iter__(self):
        return self

    def __next__(self) -> TimeSlice:
        if self._i >= len(self.__fields["times"]):
            self._i = 0
            self._totaltime = 0
            raise StopIteration

        time = self.__fields["times"].values[self._i]
        label = self.__fields["labels"].values[self._i]
        self._totaltime += time  # this isn't a tf object
        self._i += 1
        return TimeSlice(time=self._totaltime, deltat=time, label=label)

    def __repr__(self):
        return self.__fields.__repr__()


class Episodes:
    """ Contains a list of trajectories """

    def __init__(self, trajs: List[Trajectory]):
        self._trajs = trajs
        self._i = 0
        self._totaltime = functools.reduce(lambda tot, t: tot + t.totaltime, trajs, 0)
        self._nevents = dict()
        self._totalnevents = 0

        for traj in trajs:
            for label, count in traj.numevents.items():
                if label not in self._nevents:
                    self._nevents[label] = 0
                self._nevents[label] += count
                self._totalnevents += count

    @property
    def totaln(self):
        """ Contains the total count of all events """
        return self._totalnevents

    @property
    def totaltime(self):
        """ Contains the total time of all trajectories """
        return self._totaltime

    @property
    def eventcounts(self):
        """ Returns a dictionary maping labels to counts """
        return self._nevents

    def __iter__(self):
        return self

    def __next__(self) -> Trajectory:
        if self._i >= len(self._trajs):
            self._i = 0
            raise StopIteration
        self._i += 1
        return self._trajs[self._i]
