# About

This is a python package to explore point processes, specifically online learning
of temporal point process.
This is built using TensorFlow for its automatic differentiation, along with
typing to catch bugs.

So far only a multivariate Hawkes process with static base rate and a sum of
exponential triggering kernels is the only model. It has gradient ascent along
with sampling implemented.

This work is the composition of a few papers along with previous implementations
of Hawkes process, bayesian networks, and PCIMs.

## TODO

* Make the example note books
* I want an implementation of SIP along with PCIM.
* I need real data to test on.

# Tasks

A side point of this project is to compare these models to other things for
predictions. Identifying tasks is part of this project; specifically extending
these models from just temporal point processes to be used on images or in
natural language processing.

# Install

To install run

```
python setup.py install
```

An example for creating a Trajectory with 0.1 units as the discretization amount
are:

```python

import tensorflow as tf

tf.enable_eager_execution()

from Trajectory.Trajectory import Trajectory
from Trajectory.Field import Field

label_set = tf.convert_to_tensor([0, 1], dtype=tf.int32)
times = tf.convert_to_tensor([1., 2.5, 5, 9.], dtype=tf.float32)
labels = tf.convert_to_tensor([0, 1, 0, 1], dtype=tf.int32)
fields = {
    "times": Field(values=times, continuous=True, space=(0., 10.)),
    "labels": Field(values=labels, continuous=False, space=label_set)
}
trajectory = Trajectory(fields, tau=0.1)
```


An example for creating a Hawkes process, estimating parameters, and sampling
can be seen below.

```python
From Hawkes.Hawkes import Hawkes

exponential_params = [1, 3, 5]
hp = Hawkes(len(label_set), exponential_params)

hp.gradient_descent_full(trajectory, eta=0.1)

sampled_trajectory = hp.sample(max_time=10.)
```

For a more complete view along with experiments using more composable functions
please see the notebooks.
