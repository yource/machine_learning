import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
        "file_5min": "data/main5min/MA_main_5min.csv",
        "file_daily": "data/mainDaily/MA_main_daily.csv"
    },
    {
        "name":"PVC","code": "v","exchange": "DCE",
        "file_5min": "data/main5min/v_main_5min.csv",
        "file_daily": "data/mainDaily/v_main_daily.csv"
    },
    {
        "name":"聚丙烯","code": "pp","exchange": "DCE",
        "file_5min": "data/main5min/pp_main_5min.csv",
        "file_daily": "data/mainDaily/pp_main_daily.csv"
    },
    {
        "name":"豆粕","code": "m","exchange": "DCE",
        "file_5min": "data/main5min/m_main_5min.csv",
        "file_daily": "data/mainDaily/m_main_daily.csv"
    },
    {
        "name":"菜粕","code": "RM","exchange": "CZCE",
        "file_5min": "data/main5min/RM_main_5min.csv",
        "file_daily": "data/mainDaily/RM_main_daily.csv"
    },
    {
        "name":"塑料","code": "l","exchange": "DCE",
        "file_5min": "data/main5min/l_main_5min.csv",
        "file_daily": "data/mainDaily/l_main_daily.csv"
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
def getData(file):
    df = pd.read_csv(file)
    cols = df.columns
    df = df[[cols[2],cols[3],cols[4],cols[5],cols[6],cols[7],cols[8]]]
    df = df[1:]
    # 检查数据错误 最大值最小值 
    # print(df.head)
    # print(df.describe().transpose())
    # print(len(df[df[cols[6]]<1]))


getData(varieties[0]["file_5min"])

