
"""
    Author: Mehran Ghamaty

    This implementation of a PCIM that is restricted to
    change points that occur specifially on events.
    This is done as to use the scikitlearn package.
"""

from typing import Optional, List
from dataclasses import dataclass
from numpy import log, inf

import trajectory as t

@dataclass
class PCIMNode:
    """
        This is the PCIM Node
    """
    basis_function: Optional[BasisFunction]
