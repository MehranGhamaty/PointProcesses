"""
    Holds one of the fields in a trajectory.
"""

from dataclasses import dataclass, field
from typing import List, Union, Set, Tuple

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
