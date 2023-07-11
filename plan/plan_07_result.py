import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
import keras.backend as K
import keras_tuner as kt

project_name="plan_07_001"
code = "rb"; min="15"
file = "data/main/"+code+"_"+min+"min.csv"
df = pd.read_csv(file)
cols = df.columns
df = df[[cols[2],cols[3],cols[4],cols[5]]]
df = df[1:]
data = np.array(df, dtype=np.float32)
percent_train = 0.76
percent_val = 0.88
len_data = len(data)
len_train = int(len_data*percent_train)

'''
根据过去的lookback期数据 预测之后的第delay期的数据
隔step取一个有效值
隔offset个数据生成一个窗口 0代表各window不交叉
'''
print("##### 生成 样本-目标")
lookback = 12; delay = 2; step = 1; offset = 3
window_size = lookback+delay  # 每个窗口用到的数据量
if offset == 0:
    offset = window_size
window_count = (len_data-window_size)//offset  # 可以切分出多少个窗口
if(step == 0):
    step = 1
samples = np.zeros((window_count, lookback//step, data.shape[-1]))
target_percents = np.zeros(window_count)
for i in range(0, window_count):
    window = data[offset*i:offset*i+window_size]
    window_samples = window[0:lookback]
    window_mean = window_samples.mean(axis=0)
    window_samples -= window_mean
    window_std = window_samples.std(axis=0)
    window_samples /= window_std
    window_sample_value = window[lookback-1][3]
    window_target = window[lookback+delay-1]
    window_target_high = (window_target[0]+window_target[1]+window_target[3])/3
    window_target_low = (window_target[0]+window_target[2]+window_target[3])/3
    window_target_value = (((window_target[0]+window_target[1]+window_target[2])/3)+window_target[3]*2)/3
    window_target_percent = int(((window_target_value-window_sample_value)/window_sample_value)*10000)
    samples[i] = window_samples
    target_percents[i] = window_target_percent
    
n = window_count
n_train = int(percent_train * n)
n_val = int(percent_val * n)

target_percent_train = target_percents[:n_train]
target_percent_low  = np.percentile(target_percent_train, 33)
target_percent_high = np.percentile(target_percent_train, 67)
print("target_percent_low",target_percent_low)
print("target_percent_high",target_percent_high)
target_labels = np.zeros((window_count,3))

for i,y in enumerate(target_percents):
    if(y<=target_percent_low):
        target_labels[i] = [1,0,0]
    elif(y>=target_percent_high):
        target_labels[i] = [0,0,1]
    else:
        target_labels[i] = [0,1,0]

print("##### 拆分数据 训练集、验证集、测试集")
all_x = samples #约1911天的数据
all_y = target_labels
train_x = all_x[0:n_train] 
train_y = all_y[0:n_train]
val_x   = all_x[n_train:n_val]
val_y   = all_y[n_train:n_val]
test_x  = all_x[n_val:] 
test_y  = all_y[n_val:] 
print("train_x",train_x.shape)
print("val_x",val_x.shape)
print("test_x",test_x.shape)


units = 256
normal1 = True;active1 = False;dropout1 = False
normal2 = False;active2 = False;dropout2 = False
lr = 0.00034929128241395816;rate1 = 0.3;rate2 = 0.4

# units = 416
# normal1 = True;active1 = False;dropout1 = True
# normal2 = False;active2 = False;dropout2 = False
# lr = 0.0002693387023501433;rate1 = 0.45;rate2 = 0.4

# units = 448
# normal1 = True;active1 = False;dropout1 = False
# normal2 = False;active2 = True;dropout2 = False
# lr = 0.00011940882987867163;rate1 = 0.45;rate2 = 0.15

# units = 224
# normal1 = True;active1 = False;dropout1 = False
# normal2 = False;active2 = False;dropout2 = False
# lr = 0.0010145077995017421;rate1 = 0.4;rate2 = 0.25

# units = 256
# normal1 = True;active1 = False;dropout1 = False
# normal2 = False;active2 = False;dropout2 = False
# lr = 0.0012916500815288833;rate1 = 0.25;rate2 = 0.5

# units = 288
# normal1 = True;active1 = False;dropout1 = False
# normal2 = True;active2 = False;dropout2 = True
# lr = 0.001382827869491224;rate1 = 0.5;rate2 = 0.35

# units = 352
# normal1 = True;active1 = False;dropout1 = False
# normal2 = False;active2 = False;dropout2 = False
# lr = 0.00018723379392602342;rate1 = 0.15;rate2 = 0.15

# units = 448
# normal1 = True;active1 = False;dropout1 = True
# normal2 = True;active2 = False;dropout2 = False
# lr = 0.005322635881784047;rate1 = 0.1;rate2 = 0.1

# units = 96
# normal1 = True;active1 = False;dropout1 = True
# normal2 = True;active2 = False;dropout2 = False
# lr = 0.008845571743601299;rate1 = 0.15;rate2 = 0.3

# units = 160
# normal1 = True;active1 = False;dropout1 = False
# normal2 = False;active2 = True;dropout2 = True
# lr = 0.0007315903709982806;rate1 = 0.25;rate2 = 0.1

def build_model(units,normal1,active1,dropout1,normal2,active2,dropout2,lr,rate1,rate2):
    model = Sequential()
    model.add(layers.Flatten())
    model.add(keras.layers.Dense(units=units))
    if normal1:
        model.add(layers.BatchNormalization())  
    if active1: 
        model.add(layers.Activation('relu'))
    if dropout1: 
        model.add(layers.Dropout(rate=rate1))
    model.add(keras.layers.Dense(units=units))
    if normal2:
        model.add(layers.BatchNormalization())  
    if active2: 
        model.add(layers.Activation('relu'))
    if dropout2: 
        model.add(layers.Dropout(rate=rate2))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# 使用从搜索中获得的超参数找到训练模型的最佳周期数
x_all = np.concatenate((train_x, val_x))
y_all = np.concatenate((train_y, val_y))
model = build_model(units,normal1,active1,dropout1,normal2,active2,dropout2,lr,rate1,rate2)
history = model.fit(x=x_all, y=y_all, epochs=500, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# 重新实例化超模型并使用上面的最佳周期数对其进行训练
hypermodel = build_model(units,normal1,active1,dropout1,normal2,active2,dropout2,lr,rate1,rate2)
hypermodel.fit(x_all, y_all, epochs=best_epoch)
# 保存模型
hypermodel.save("model/"+project_name+"_01.h5")
# 重新加载模型
# newModel = keras.models.load_model("./model/mnist")
# print(newModel.summary())

predictions = model.predict(test_x)
testLen = len(test_y)
test = 0
test_y_num = np.zeros(testLen)
pred_y_num = np.zeros(testLen)
pred = 0
pred_right = 0
pred_wrong = 0
# 0负 1平 2盈
for i,y in enumerate(test_y):
    test_y_num[i] = np.argmax(y)
for i,y in enumerate(predictions):
    if(np.argmax(y)==0 and y[0]>0.4 and y[2]<0.3):
        pred_y_num[i] = 0
    elif(np.argmax(y)==2 and y[2]>0.4 and y[0]<0.3):
        pred_y_num[i] = 2
    else:
        pred_y_num[i] = 1

for i in range(0,testLen):
    if(test_y_num[i] != 1):
        test += 1
        # print("test y", test_y[i])
        # print("pred", predictions[i])
    if(pred_y_num[i] != 1):
        pred += 1
        if(test_y_num[i] == pred_y_num[i]):
            pred_right += 1
        else:
            pred_wrong += 1

print(test,pred,pred_right,pred_wrong)