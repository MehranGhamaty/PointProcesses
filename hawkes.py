"""
    Class for a hawkre is process.
    Assumes that the session is executing eagerly.
"""
from typing import List
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from trajectory import Trajectory
from kernels import Kernel, State

class Hawkes:
    """
	A general multivariate Hawkes process that uses a sum of exponentials
    """
    def __init__(self, nlabels: int, kernel: Kernel):
        self._kernel: Kernel = kernel
        self._weights = tfe.Variable(
            initial_value=tf.random.uniform((nlabels, nlabels),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)
        self._mus = tfe.Variable(
            initial_value=tf.random.uniform((nlabels,),
                                            minval=0.001, maxval=1),
            dtype=tf.float32, trainable=True)
        self._states: List[State] = [State() for _ in range(nlabels)]

    def __repr__(self):
        return "{}\n{}\n".format(self._weights, self._mus)

    def sample(self, end_time) -> Trajectory:
        """
	    :param end_time: The end time of the sample
	    :return: The sampled trajectory
	"""
        pass

    def discretellh(self, traj: Trajectory) -> float:
        """
            :param traj: The trajectory to calculate the llh of
        """
        llh: float = 0.
        for segval in self.calcsegllh(traj):
            for score in segval.values():
                llh += score

        return llh

    def calcsegllh(self, traj: Trajectory) -> tfe.Variable:
        """
            :param traj: The trajectory to calculate the log likelihood of
            :return: The log likelihood.
        """
        for label in traj.field["labels"].space:
            self._states[label] = State()

        for time, delta, evlabel in traj:
            segscore = tfe.Variable(tf.zeros((len(traj.field["labels"]),)))

            for label in traj.field["labels"].space:
                self._kernel.advancestate(self._states[label], time)
                self._kernel.addevent(self._states[label], self._weights[evlabel][label])

            print("finished adv")
            for label in traj.field["labels"].space:
                print("lab", label)
                print("states", self._states[label])
                print("segscore", segscore[label])
                change = delta * (self._states[label].intensity + self._mus[label])
                if evlabel != -1:
                    change -= tf.math.log(self._states[label].intensity + self._mus[label])
                segscore = segscore[label].assign(segscore[label] + change)

            yield segscore

    def sgd(self, eta: float, traj: Trajectory) -> None:
        """
            Performs stochastic gradient descent.
        """
        for time, delta, evlabel in traj:
            segscore = tfe.Variable(0, name='segment_score', dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                #I'm going to need the dmu[i] and d_w[i,j]
                tape.watch(segscore)
                for label in traj.field["labels"].space:
                    self._kernel.advancestate(self._states[label], time)
                    self._kernel.addevent(self._states[label], self._weights[evlabel][label])
                print("finished adv")
                for label in traj.field["labels"].space:
                    print("lab", label)
                    print("states", self._states[label])
                    print("segscore", segscore)
                    change = delta * (self._states[label].intensity + self._mus[label])
                    if evlabel != -1:
                        change -= tf.math.log(self._states[label].intensity + self._mus[label])
                    print("delta", change)
                    segscore.assign(segscore+change)
            print(segscore)
            print(tape.gradient(segscore, self._mus))
            for labeli in traj.field["labels"].space:
                dmu_li = tape.gradient(segscore, self._mus[labeli])
                self._mus[labeli] -= eta * dmu_li

                for labelj in traj.field["labels"].space:
                    dw_lilj = tape.gradient(segscore, self._weights[labeli, labelj])
                    self._weights[labeli, labelj] -= eta *  dw_lilj

            del tape
