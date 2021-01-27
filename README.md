# Description


The python package is written to get publications from point processes.
The toolkit is written with TensorFlow's automatic differentiation, along with
strict typing to catch bugs 
(which means if a function asks for something specifically only that type of object can be provided as a parameter).
 A multivariate Hawkes process with non-variable base rate and a sum of
exponential triggering kernels is the only model currently implemented. 

One of the papers that this work was attempting verify was a Hawkes process estimating the rate at which patients enter or exit a hospital. The task being scheduling avaliability of beds. Another task was modeling survival rates, if done in real-time it could help with providing IV bags.

# Install

To install run

```
python setup.py install
```

An example for creating a Trajectory with 0.1 units as the discretization amount
is given below: 

```python

import tensorflow as tf

tf.enable_eager_execution()

from DataStores.Trajectory import Trajectory, Field

label_set = tf.convert_to_tensor([0, 1], dtype=tf.int32)
times = tf.convert_to_tensor([1., 2.5, 5, 9.], dtype=tf.float32)
labels = tf.convert_to_tensor([0, 1, 0, 1], dtype=tf.int32)
fields = {
    "times": Field(values=times, continuous=True, space=(0., 11.)),
    "labels": Field(values=labels, continuous=False, space=label_set)
}
trajectory = Trajectory(fields, tau=0.1)
```
Replacing 0.1 with ```np.inf``` will have the process learn in continuous time, which generates a higher log likihood with lower computational cost, in specific situations.

An example for creating a Hawkes process, estimating parameters, and sampling
can be seen below.

```python
From PointProcesses.Hawkes import Hawkes

exponential_params = [1, 3, 5]
hp = Hawkes(len(label_set), exponential_params)

hp.gradient_descent_full(trajectory, eta=0.1)

sampled_trajectory = hp.sample(max_time=12.)
```

