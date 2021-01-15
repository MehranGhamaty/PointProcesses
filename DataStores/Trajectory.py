"""
    A class to hold trajectories of events that contain labels.
    used for online streaming. Contains classes for a TimeSlice,
    Field, Trajectory, and Episodes.


    A TimeSlince  contains the real time stamp, the time since the last event,
    a label and a mark.

    A Field is one of the members in a Trajectory.
 
    A Trajectory manages a set of Fields.

    An Episodes object manages a set of Trajectories.

    The label should actually be an optional attribute,

    On the side lets do the lat long representation as well,
"""
import functools
from dataclasses import dataclass, field

from typing import Dict, List, Tuple, Union, Set, Optional, String
from numpy import inf

import tensorflow as tf


@dataclass
class TimeSlice:
    """
        Each of these serves as an example for the dataset.
    """
    time: tf.constant  # the end time of the slice
    deltat: tf.constant  # the duration of the slice
    mark: Optional[Dict[String, tf.constant]] = None


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
        This is for online learning, meaning that the delta is the minimum
        amount of time where no events occur before an update takes place.
   """

    def __init__(self, fields: Dict[str, Field], tau: float = inf):
        """
            Here we will note that the times are now going to be paritioned
            such
            that t_0 = 0, and the following will be the next time or t_{k-1} +
            \tau,
            which ever comes first. If t_{k-1} + \tau is
            first then the associated label is None.

            The issue being when I'm going to calculate the sums
            I'm going to do a sum over the labels.

            Assumes that there will be at most 3 fields; times, labels, and
            It shouldn't matter what I call the fields except for time.

            :param fields: A dictionary containing the fields
            :param delta: the discretization period.
        """

        # tau = tf.convert_to_tensor(tau, dtype=tf.float32)
        # neglab = tf.convert_to_tensor(-1, dtype=tf.int32)

        self._numevents = dict()
        self._totaltime = fields["times"].space[1] - fields["times"].space[0]

        self._fields = dict()
        for key, val in fields.items():
            self._fields[key] = Field(continuous=val.continuous,
                                      space=val.space)

        currtime = self._fields["times"].space[0]
        for ind, time in enumerate(fields["times"].values):

            timeuntilnextevent = time - currtime
            while timeuntilnextevent > tau:
                timeuntilnextevent -= tau
                self._addevent((tau, -1))

            label = fields["label"].values[ind]
            if label not in self.numevents:
                self.numevents[label] = 0
            self.numevents[label] += 1

            # for each other field I need to add its info to addevent
            aux = {}
            for key in fields.keys():
                if key != "times" or key != "labels":
                    aux[key] = fields[key].values[ind]

            if timeuntilnextevent is tf.Tensor:
                self._addevent((timeuntilnextevent.numpy(), label.numpy()), **aux)
            else:
                self._addevent((timeuntilnextevent, label), **aux)

        # Convert everything to tensors
        for key in fields.keys():
            self._fields[key].values = tf.convert_to_tensor(
                self._fields[key].vaues,
                dtype=tf.float32 if self._fields[key].continuous else tf.int32)

        # Used for streaming
        self._totaltime = 0
        self._i = 0

    def resetstate(self):
        """ Resets the total time for iterating over """
        self._totaltime = 0
        self._i = 0

    @property
    def totaltime(self):
        """ returns the total amount of time in the traj """
        return self._totaltime

    @property
    def numevents(self):
        """ returns a dictionary that maps the labs to number of events """
        return self._numevents

    def add_time_slice(self, time_slice: TimeSlice):
        """
            No validation checks pretty sure this is not going
            to work if I have none types.
        """
        tf.stack([self._fields["times"].values.numpy(),
                  [time_slice.deltat.numpy()]])
        tf.stack([self._fields["labels"].values.numpy(),
                  [time_slice.label.numpy()]])

    def _addevent(self, event: Tuple[float, int], **kwargs):
        """
            Adds an event to the trajectory
            For use during init
        """
        self._fields["times"].values.append(event[0])
        self._fields["labels"].values.append(event[1])

        for key, arg in kwargs.items():
            self._fields[key].values.append(arg)

    @property
    def field(self) -> Dict[str, Field]:
        """ Returns the fields """
        return self._fields

    def __len__(self) -> int:
        return len(self._fields["times"])

    def __iter__(self):
        return self

    def __next__(self) -> TimeSlice:
        if self._i >= len(self._fields["times"]):
            self._i = 0
            self._totaltime = 0
            raise StopIteration

        time = self._fields["times"].values[self._i]
        label = self._fields["labels"].values[self._i]
        self._totaltime += time  # this isn't a tf object
        self._i += 1
        return TimeSlice(time=self._totaltime, deltat=time, label=label)

    def __repr__(self):
        return self._fields.__repr__()


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
