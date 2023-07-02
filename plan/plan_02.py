'''
PLAN_01
1. 只用 open/high/low/close 数据
2. 不做归一化 
3. 分类 涨-跌-平
------
甲醇_MA VC_v 聚丙烯_pp 豆粕_m 菜粕_RM 塑料_l
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import layers
from keras.models import Sequential

code = "MA"; min="15"
modelName = "model/plan02_01_"+code+min+".h5"
file = "data/main/"+code+"_"+min+"min.csv"
df = pd.read_csv(file)
cols = df.columns
df = df[[cols[2],cols[3],cols[4],cols[5]]]
df = df[1:]
df = np.array(df, dtype=np.float32)
percent_train = 0.7
percent_val = 0.85

'''
根据过去的lookback期数据 预测之后的第delay期的数据
隔step取一个有效值
隔offset个数据生成一个窗口 0代表各window不交叉
'''
def generator(data, lookback, delay, step=1, offset=0):
    data_len = len(data)
    window_size = lookback+delay  # 每个窗口用到的数据量
    if offset == 0:
        offset = window_size
    window_count = (data_len-window_size)//offset  # 可以切分出多少个窗口
    if(step == 0):
        step = 1
    samples = np.zeros((window_count, lookback//step, data.shape[-1]))
    
    targets = np.zeros((window_count,3))
    for i in range(0, window_count):
        window = data[offset*i:offset*i+window_size]
        window_samples = window[0:lookback]
        window_sample_value = window[lookback-1][3]
        window_target = window[lookback+delay-1]
        window_target_high = (window_target[0]+window_target[1]+window_target[3])/3
        window_target_low = (window_target[0]+window_target[2]+window_target[3])/3
        window_label = np.array([0,1,0]) #平
        if(window_target_low>window_sample_value and window_target_high>window_sample_value*1.01):
                window_label = np.array([0,0,1]) #涨
        elif(window_target_low<window_sample_value*0.99 and window_target_high<window_sample_value):
                window_label = np.array([1,0,0]) #跌
        
        samples[i] = window[0:lookback:step]
        targets[i] = window_label
    return samples, targets

print("##### 生成 样本-目标")
lookback = 6; delay = 2; step = 1; offset = 3
all_x, all_y = generator(df,lookback=lookback,delay=delay,step=step,offset=offset)
print("all_x",all_x.shape)
print("all_y",all_y.shape)
print("##### 拆分数据 训练集、验证集、测试集")
n = len(all_x)
n_train = int(percent_train * n)
n_val = int(percent_val * n)
train_x = all_x[0:n_train]  #约1911天的数据
val_x = all_x[n_train:n_val]
test_x = all_x[n_val:] 
train_y = all_y[0:n_train]
val_y = all_y[n_train:n_val]
test_y = all_y[n_val:] 
print("train_x",train_x.shape)
print("val_x",val_x.shape)
print("test_x",test_x.shape)

model = Sequential([
    # layers.LSTM(32, input_shape=(None,train_x.shape[-1]) ,return_sequences=True),
    layers.LSTM(32, input_shape=(None,train_x.shape[-1])),
    layers.Dense(3, activation='softmax')
])
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
# metrics=[keras.metrics.MeanAbsoluteError()]
# callbacks=[early_stopping]
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history = model.fit(train_x,train_y, 
                    epochs=10, batch_size=256,
                    validation_data=(val_x,val_y))

# 绘制损失曲线和精度曲线
def showLossAcc(_history):
    loss = _history['loss']
    val_loss = _history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    plt.clf()
    acc = _history['accuracy']
    val_acc = _history['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# showLossAcc(history.history)

# results = model.evaluate(test_x, test_y)
# print("results--")
# print(results)

predictions = model.predict(test_x)
testLen = len(test_y)
test = 0
test_y_num = np.zeros(testLen)
pred_y_num = np.zeros(testLen)
pred = 0
pred_right = 0
pred_wrong = 0
for i,y in enumerate(test_y):
    test_y_num[i] = np.argmax(y)
for i,y in enumerate(predictions):
    pred_y_num[i] = np.argmax(y)
for i in range(0,testLen):
    if(test_y_num[i] != 1):
        test += 1
    if(pred_y_num[i] != 1):
        pred += 1
        if(test_y_num[i] == pred_y_num[i]):
            pred_right += 1
        else:
            pred_wrong += 1

print(test,pred,pred_right,pred_wrong)

# xar = range(1, len(test_y_num)+1)
# plt.plot(xar, test_y_num, 'bo', label='Test Y')
# plt.plot(xar, pred_y_num, 'b', label='Pred Y')
# plt.legend()
# plt.show()

# model.save(modelName)

