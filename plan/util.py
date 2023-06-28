import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 作图，看历史走势
def makeFig(file,filePic):
    df = pd.read_csv(file)
    cols = df.columns 
    plot_features = df[cols[5]]
    plot_date = pd.to_datetime(df.pop('datetime'), format='%Y-%m-%d')
    plot_features.index = plot_date
    plot = plot_features.plot()
    plt.show()
    fig = plot.get_figure()
    fig.savefig(filePic)

'''
根据过去的lookback期数据 预测之后的第delay期的数据
隔step取一个有效值
隔offset个数据生成一个窗口 0代表各window不交叉
'''
def generator(data, lookback, delay, step=1, offset=0,normalize=None):
    data_len = len(data)
    window_size = lookback+delay  # 每个窗口用到的数据量
    if offset == 0:
        offset = window_size
    window_count = (data_len-window_size)//offset  # 可以切分出多少个窗口
    if(step == 0):
        step = 1
    samples = np.zeros((window_count, lookback//step, data.shape[-1]))
    targets = np.zeros((window_count,))
    for i in range(0, window_count):
        window = data[offset*i:offset*i+window_size]
        window_samples = window[0:lookback]
        window_sample_value = window[lookback-1][3]
        window_target = window[lookback+delay-1]
        window_target_high = (window_target[0]+window_target[1]+window_target[3])/3
        window_target_low = (window_target[0]+window_target[2]+window_target[3])/3
        window_label = 0
        if(window_target_low>window_sample_value and window_target_high>window_sample_value*1.01):
            window_label = 1
        elif(window_target_low<window_sample_value*0.99 and window_target_high<window_sample_value):
            window_label = -1
        
        if(normalize=='SMA'):
            window_mean = window_samples.mean(axis=0)
            window_std = window_samples.std(axis=0)
            for ii in range(0, 4):
                if(window_std[ii] == 0):
                    window_std[ii] = 1
            window = (window-window_mean)/window_std
        
        samples[i] = window[0:lookback:step]
        # targets[i] = window[lookback+delay-1][0]
        targets[i] = window_label
    return samples, targets


# 读取文件 处理数据
# normalize: None Simple SMA简单移动平均 WMA权重易懂平均 EMA指数移动平均
def getData(code, min, normalize):
    file = "data/main/"+code+"_"+str(min)+"min.csv"
    df = pd.read_csv(file)
    cols = df.columns
    df = df[[cols[2],cols[3],cols[4],cols[5]]]
    df = df[1:]
    # print("##### 检查数据错误 最大值最小值")
    # print(df.head)
    # print(df.describe().transpose())
    
    df = np.array(df, dtype=np.float32)
    n = len(df)

    if(normalize=='Simple'):
        print("##### 数据归一化 简单平均数")
        data_train = df[0: int(0.75 * n)]
        train_mean = data_train.mean(axis=0)
        train_std = data_train.std(axis=0)
        df = (df-train_mean)/train_std

    print("##### 拆分数据 训练集、验证集、测试集")
    data_train = df[0: int(0.7 * n)]  #约1911天的数据
    data_val = df[int(0.7 * n):int(0.88 * n)]
    data_test = df[int(0.88 * n):]

    print("##### 生成 样本-目标")
    lookback = 6
    delay = 2
    step = 1
    offset = 2
    train_x, train_y = generator(data_train,lookback=lookback,delay=delay,step=step,offset=offset,normalize=normalize)
    val_x, val_y = generator(data_val,lookback=lookback,delay=delay,step=step,offset=offset,normalize=normalize)
    test_x, test_y = generator(data_test,lookback=lookback,delay=delay,step=step,offset=offset,normalize=normalize)
    # len1 = len(np.where(train_y<0)[0])
    # len2 = len(np.where(train_y>0)[0])
    # print(str(len1)+"+"+str(len2)+"="+str(len1+len2))
    return (train_x, train_y),(val_x, val_y),(test_x, test_y)

