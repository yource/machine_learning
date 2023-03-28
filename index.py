import pymongo
import numpy as np
import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

dbclient = pymongo.MongoClient('mongodb://localhost:27017/')
db = dbclient['ssc']
collection = db['hn5fcs']
finds = collection.find()
count = finds.count()
print("count: ",count)
data = np.zeros((count,5))

for i,r in enumerate(finds):
    data[i] = [float(int(x)/10) for x in r['code'].split(",")]
    
pre = 244
exp = 44
oneCount = pre+exp 
length = count//oneCount
train_len = int(length*0.75) # 训练的数据量
train_samples = np.zeros((train_len,pre,5))
train_targets = np.zeros((train_len,exp,5))
test_len = length-train_len # 测试的数据量
test_samples = np.zeros((test_len,pre,5))
test_targets = np.zeros((test_len,exp,5))
i=0
while i<train_len+test_len:
    if i<train_len :
        train_data = data[i*oneCount:(i+1)*oneCount]
        train_samples[i] = train_data[0:pre]
        train_targets[i] = train_data[pre:]
    else:
        test_data = data[i*oneCount:(i+1)*oneCount]
        test_samples[i-train_len] = test_data[0:pre]
        test_targets[i-train_len] = test_data[pre:]
    i += 1

print("train_samples====")
print(train_samples.shape)
print(train_samples[0][0])
print("train_targets====")
print(train_targets.shape)
print(train_targets[train_len-1][exp-1])
print("test_samples====")
print(test_samples.shape)
print(test_samples[0][0])
print("test_targets====")
print(test_targets.shape)
print(test_targets[test_len-1][exp-1])

#### generate labels
def listSort(item):
    return item['count']
def getMaxNums(_list, position, maxNumLen): 
    count = np.zeros((10))
    num_count = [
        { 'num':0, 'count':0 },
        { 'num':1, 'count':0 },
        { 'num':2, 'count':0 },
        { 'num':3, 'count':0 },
        { 'num':4, 'count':0 },
        { 'num':5, 'count':0 },
        { 'num':6, 'count':0 },
        { 'num':7, 'count':0 },
        { 'num':8, 'count':0 },
        { 'num':9, 'count':0 },
    ]
    for m,item in enumerate(_list):
        num_count[int(item[position]*10)]['count'] += 1
    num_count.sort(key=listSort,reverse=True)
    if maxNumLen>1 :
        for i in range(maxNumLen):
            count[num_count[i]['num']] = 1
        return count
    else:
        return num_count[0]['num']/10


# postion=2 max_count=3
train_labels = np.zeros((train_len))
for i,_list in enumerate(train_targets):
    train_labels[i] = getMaxNums(_list,2,1)
print("train_labels====")
print(train_labels.shape)
print(train_labels[train_len-1])
test_labels = np.zeros((test_len))
for i,_list in enumerate(test_targets):
    test_labels[i] = getMaxNums(_list,2,1)
print("test_labels====")
print(test_labels.shape)
print(test_labels[test_len-1])


model = Sequential()
model.add(layers.LSTM(128, 
                     return_sequences=True,
                     input_shape=(None,5)))
model.add(layers.LSTM(256,
                     activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', 
              loss='mae',
              metrics=['accuracy'])
history = model.fit(train_samples,
                    train_labels,
                    epochs=50,
                    batch_size=128,
                    validation_split=0.2)

his = history.history
loss = his['loss']
val_loss = his['val_loss']
axis = range(1, len(loss)+1)
plt.plot(axis, loss, 'bo', label="Training loss")
plt.plot(axis, val_loss, 'b', label="Validation loss")
plt.legend()
plt.show()