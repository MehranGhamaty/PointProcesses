"""
    Time Slice class holds a discrete portion of time.
"""

from dataclasses import dataclass

import tensorflow as tf

@dataclass
class TimeSlice:
    """
        Each of these serves as an example for the dataset.
    """
    time: tf.constant  # the end time of the slice
    deltat: tf.constant  # the duration of the slice
    label: tf.constant  # if -1 no event occured
