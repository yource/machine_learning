import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 读取文件
file_c = './data/futuresDaily/C9999.XDCE.csv' # 玉米
file_cs = './data/futuresDaily/CS9999.XDCE.csv' # 淀粉
file_sr = './data/futuresDaily/SR9999.XZCE.csv' # 白糖
# 设置浮点数精度
pd.set_option("display.float_format", "{:.6f}".format) 
# 清理空值
dataframe = pd.read_csv(file_sr,na_values = ["n/a", "na", "--",""])
df = dataframe.dropna()

# 整体作图
# plot_cols = ['open','close']
# plot_features = df[plot_cols]
# date = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
# plot_features.index = date
# print(plot_features.shape)
# print(plot_features.head)
# plot_features.plot()
# plt.show()

# 选取数据和特征工程 将日期转为一个周期中的某一天/一年中的某一天
symbols = df.pop('symbol')
df = df[['open','close','high','low','volume','open_interest','money']]
n = len(df)
cur_symbol = ""
cur_symbol_len = 0
feature_date = np.zeros(n)
symbol_len_list = []
for i,sy in enumerate(symbols):
    if(sy!=cur_symbol): 
        cur_symbol = sy
        if(cur_symbol_len>0):
            symbol_len_list.append(cur_symbol_len)
        cur_symbol_len = 1
    else: 
        cur_symbol_len += 1
lenlist = np.array(symbol_len_list)
print("symbol_len_list",symbol_len_list)
print("n",n)
print(lenlist.sum())
print(len(lenlist))

#拆分数据
# train_len = int(n*0.7)
# val_len = int(n*0.9)
# test_len = n-train_len-val_len
# train_df = df[0:train_len]
# val_df = df[train_len:val_len]
# test_df = df[val_len:]
# num_features = df.shape[1]
# print(num_features)