# Description


The python package is written to get publications from processes.
The toolkit is written with TensorFlow's differentiation, along with
strict typing to prevent bugs 
(which means if a function asks for something specifically only that type of object can be provided as a parameter).
 A multivariate Hawkes process with non-variable base measure.

# Install
```
python setup.py install
```

# Trajectory 

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

Replace ```0.1```   with ```np.inf```  to make the process learn in continuous time, which can result in a higher log likihood with lower computational cost.

# Learning and Sampling

An example for creating a Hawkes process, finding parameters, and sampling
can be seen below.

```python
From PointProcesses.Hawkes import Hawkes

exponential_params = [1, 3, 5]
hp = Hawkes(len(label_set), exponential_params)

hp.gradient_ascent_full(trajectory, eta=0.1)

sampled_trajectory = hp.sample(max_time=12.)
```
Learning from real data can be accomplished 
simply.
