import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import util

'''
分析六大品种
甲醇	MA	   郑商所 CZCE
PVC	   v	  大商所 DCE 
聚丙烯	pp	   大商所 DCE
豆粕	m	   大商所 DCE
菜粕	RM	   郑商所 CZCE
塑料	l	   大商所 DCE	

cols 表格列 
2:open 3:high 4:low 5:close 6:volume 7:open_oi 8:close_oi
datetime: K线起点时间(按北京时间) 自unix epoch(1970-01-01 00:00:00 GMT)以来的纳秒数)
open: K线起始时刻的最新价
high: K线时间范围内的最高价
low: K线时间范围内的最低价
close: K线结束时刻的最新价
volume: K线时间范围内的成交量
open_oi: K线起始时刻的持仓量
close_oi: K线结束时刻的持仓量
'''

varieties = [
    {
        "name":"甲醇", "code": "MA","exchange": "CZCE",
    },
    {
        "name":"PVC","code": "v","exchange": "DCE",
    },
    {
        "name":"聚丙烯","code": "pp","exchange": "DCE",
    },
    {
        "name":"豆粕","code": "m","exchange": "DCE",
    },
    {
        "name":"菜粕","code": "RM","exchange": "CZCE",
    },
    {
        "name":"塑料","code": "l","exchange": "DCE",
    }
]

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

def makeFigs():
    for m,item in enumerate(varieties):
        makeFig(item["file_5min"],'pics/main5min/'+item["code"]+'.png')
        makeFig(item["file_daily"],'pics/mainDaily/'+item["code"]+'.png')

# 读取文件 处理数据
def getData(code, min):
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
    # print("##### 数据归一化 减去平均值、除以标准差")
    # data_train = df[0: int(0.75 * n)]
    # train_mean = data_train.mean(axis=0)
    # train_std = data_train.std(axis=0)
    # df = (df-train_mean)/train_std

    print("##### 拆分数据 训练集、验证集、测试集")
    data_train = df[0: int(0.7 * n)]  #约1911天的数据
    data_val = df[int(0.7 * n):int(0.88 * n)]
    data_test = df[int(0.88 * n):]

    print("##### 生成 样本-目标")
    lookback = 6
    delay = 2
    step = 1
    offset = 2
    train_x, train_y = util.generator(data_train,lookback=lookback,delay=delay,step=step,offset=offset)
    val_x, val_y = util.generator(data_val,lookback=lookback,delay=delay,step=step,offset=offset)
    test_x, test_y = util.generator(data_test,lookback=lookback,delay=delay,step=step,offset=offset)
    # len1 = len(np.where(train_y<0)[0])
    # len2 = len(np.where(train_y>0)[0])
    # print(str(len1)+"+"+str(len2)+"="+str(len1+len2))
    # print(str(len1+len2)+"/"+str(len(train_x))+"="+str((len1+len2)/len(train_x)))
    # print(1911/(len1+len2))
    print(train_x.shape)
    print(train_y.shape)
    print(val_x.shape)
    print(val_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    
    


getData(varieties[0]["code"],20)
