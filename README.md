# Description

The python package is written to explore point processes, specifically online learning
of temporal point process.
The tool is constructed using TensorFlow, using automatic differentiation, along with
static typing to catch bugs.

A multivariate Hawkes process with static base rate and a sum of
exponential triggering kernels is the only model currently implemented. 

This work is the composition of a few papers along with previous implementations
of Hawkes process, bayesian networks, and PCIMs.

# Install

To install run

```
python setup.py install
```

An example for creating a Trajectory with 0.1 units as the discretization amount
are (If you want continuous set it to np.inf, it will be much faster to run and 
the estimated parameters will have a higher likelihood):

```python

import tensorflow as tf

tf.enable_eager_execution()

from DataStores.Trajectory import Trajectory, Field

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
From PointProcesses.Hawkes import Hawkes

exponential_params = [1, 3, 5]
hp = Hawkes(len(label_set), exponential_params)

hp.gradient_descent_full(trajectory, eta=0.1)

sampled_trajectory = hp.sample(max_time=10.)
```

For a more complete view along with experiments using more composable functions
please see the notebooks. Keep in mind that most of this code base is fairly useless,
it was an attempt at writing a generalized toolkit for temporal point processes.

I chose to use GD for online learning because it is the most efficient when the dataset
size reaches infinity (I should find the citation for that).

The problem is that shown in one of the notebooks, a Hawkes process with a single 
exponential kernel should have a loss function with a global minimum. This means
when sampling from the model and relearning a new model from the sampled model 
the parameters should converge. That wasn't happening, I know the code is working
because I verified all the parts. This means there is probably something wrong with my 
mathematical assumptions and I want to rewrite the code using EM to verify above statement.

Yes research is hard and annoying and this is why graduate students not working with existing toolkits 
take longer to produce research.
