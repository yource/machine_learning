import pymongo
import numpy as np

dbclient = pymongo.MongoClient('mongodb://localhost:27017/')
db = dbclient['ssc']
collection = db['hn5fcs']
finds = collection.find()
count = finds.count()
print("count: ",count)
data = np.zeros((count,5))

for i,r in enumerate(finds):
    data[i] = [float(int(x)/10) for x in r['code'].split(",")]
    
pre = 244 # 根据过去的多少期
exp = 44  # 预测接下来的多少期
oneCount = pre+exp 
length = count//oneCount
train_len = int(length*0.8) # 训练的数据量
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

