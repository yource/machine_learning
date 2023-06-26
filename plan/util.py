import numpy as np

'''
根据过去的lookback期数据 预测delay期之后的数据
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
    targets = np.zeros((window_count,))
    for i in range(0, window_count):
        window = data[offset*i:offset*i+window_size]
        window_samples = window[0:lookback]
        window_mean = window_samples.mean(axis=0)
        window_std = window_samples.std(axis=0)
        for ii in range(0, 4):
            if(window_std[ii] == 0):
                window_std[ii] = 1
        window = (window-window_mean)/window_std
        samples[i] = window[0:lookback:step]
        targets[i] = window[lookback+delay-1][0]
    return samples, targets