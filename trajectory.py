"""
	A class to hold trajectories of events that contain labels.
	used for online streaming.
"""
from typing import Dict, Union, List, Set, Tuple
from dataclasses import dataclass, field
from numpy import inf

@dataclass
class TimeSlice:
    """
        Each of these serves as an example for the dataset.
    """
    time: float #the end time of the slice
    deltat: float # the duration of the slice
    label: int # if -1 no event occured

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
        Here we will note that the times are now going to be paritioned such that t_0 = 0,
        and the following will be the next time or t_{k-1} + \tau, which ever comes first.
        If t_{k-1} + \tau is first then the associated label is -1

        The issue being when I'm going to calculate the sums I'm going to do a sum over the labels

        :param delta: the discretization period.
        """
#assert all(times[i] <= times[i+1] for i in range(len(times)-1)), "The times are not sorted!"
#assert len(times) == len(labels), "Number of items in times and labels differ!"
#assert times[0] >= duration[0], "There are events that occur before the time range"
#assert times[-1] <= duration[1], "There are events that occur after the range"

        self.__fields = dict()
        for key, val in fields.items():
            self.__fields[key] = Field(continuous=val.continuous, space=val.space)

        currtime = self.__fields["times"].space[0]
        for time, label in zip(fields["times"].values, fields["labels"].values):
            timeuntilnextevent = time - currtime
            while timeuntilnextevent > tau:
                timeuntilnextevent -= tau
                self._addevent((tau, -1))
            self._addevent((timeuntilnextevent, label))
        self._totaltime = 0
        self._i = 0

    def _addevent(self, event: Tuple[float, int]):
        """ Adds an event to the trajectory """
        self.__fields["times"].values.append(event[0])
        self.__fields["labels"].values.append(event[1])

    @property
    def field(self) -> Dict[str, Field]:
        """ Returns the fields """
        return self.__fields

    def __iter__(self):
        return self
    def __next__(self) -> TimeSlice:
        if self._i >= len(self.__fields["times"]):
            raise StopIteration
        else:
            time = self.__fields["times"].values[self._i]
            label = self.__fields["labels"].values[self._i]
            self._totaltime += time
            self._i += 1
            return TimeSlice(time=self._totaltime, deltat=time, label=label)

    def __repr__(self):
        return self.__fields.__repr__()
