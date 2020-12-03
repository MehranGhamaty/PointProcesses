

import pickle


folder = '../Datasets/competitive-data-science-predict-future-sales'
trainingtimesfile = '{}/train.times.pickle'.format(folder)
traininglabelsfile = '{}/train.labels.pickle'.format(folder)
traininglabelmapfile = '{}/train.labelmap.pickle'.format(folder)

with open(trainingtimesfile, 'rb') as f:
    times = pickle.load(f)
with open(traininglabelsfile, 'rb') as f:
    labels = pickle.load(f)
with open(traininglabelmapfile, 'rb') as f:
    labelmap = pickle.load(f)

import tensorflow as tf
start = times[0]
end = times[-1]
times = tf.convert_to_tensor(times, type=tf.float32)
labels = tf.convert_to_tensor(labels, type=tf.int32)
labelset = tf.convert_to_tensors(list(labelmap.values()), type=tf.float32)

traj = Trajectory(
    {
        "times" : Field(values=times, continuous=True, space=(start, end)),
        "labels" : Field(values=labels, continuous=False, space=labelset)
    })
 
