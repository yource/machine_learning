import pymongo
import numpy as np

dbclient = pymongo.MongoClient('mongodb://localhost:27017/')
db = dbclient['ssc']
collection = db['hn5fcs']
records =collection.find()
count = records.count()
print("count: ",count)
data = np.zeros((count,5))

for i,r in enumerate(records):
    data[i] = [float(int(x)/10) for x in r['code'].split(",")]

position = [0] #要预测的位置
pre = 244 # 根据过去的多少期
exp = 44  # 预测接下来的多少期
oneCount = pre+exp 
len = count//oneCount
train_len = int(len*0.8) # 训练的数据量
train_samples = np.zeros((train_len,pre,5))
train_targets = np.zeros((train_len,exp,5))
test_len = int(len*0.2) # 测试的数据量
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
        test_samples[i] = test_data[0:pre]
        test_targets[i] = test_data[pre:]
    i += 1

print("train_samples====")
print(len(train_samples))
print(train_samples[0])
print(train_samples[len(train_samples)-1])
print("test_samples====")
print(len(test_samples))
print(test_samples[0])
print(test_samples[len(test_samples)-1])
