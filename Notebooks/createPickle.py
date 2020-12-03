import gzip
from datetime import datetime
import pickle

folder = '../Datasets/competitive-data-science-predict-future-sales/'
training = folder + 'sales_train.csv.gz'
testing = folder + 'test.csv.gz'
outputlabels = folder + 'train.labels.pickle'
outputtimes = folder + 'train.times.pickle'
outputlabelmap = folder + 'train.labelmap.pickle'

labelmap = {}
reverselabelind = {}
labeli = 0

dateformat = '%d.%m.%Y'

times = []
labels = []

numlines = 0
endtime = 0

with gzip.open(training, 'rb') as f:
    header = f.readline()
    print(header)
    for line in f:
        tokens = line.decode("utf-8").split(',')
        date = datetime.strptime(tokens[0], dateformat).timestamp()
        shopid = int(tokens[2])
        itemid = int(tokens[3])
        itemprice = float(tokens[4])
        itemcount = int(round(float(tokens[5].strip())))
        
        pair = (shopid, itemid)
        
        if pair not in labelmap:
            labelmap[pair] = labeli
            reverselabelind[labeli] = pair
            labeli += 1
        
        for _ in range(itemcount):
            times.append(date)
            labels.append(labelmap[pair])
            
        numlines += 1
        
        if date > endtime:
            endtime = date
        
        print(numlines)

times, labels = zip(*sorted(zip(times, labels), key=lambda x: x[0]))
#times = tf.convert_to_tensor(times, dtype=tf.float32)
#labels = tf.convert_to_tensor(labels, dtype=tf.int32)

with open(outputtimes, 'wb') as f:
    pickle.dump(times, f)
with open(outputlabels, 'wb') as f:
    pickle.dump(labels, f)
with open(outputlabelmap, 'wb') as f:
    pickle.dump(labelmap, f) 
