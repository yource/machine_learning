import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import util

"""
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
"""

varieties = [
    {"name": "甲醇", "code": "MA", "exchange": "CZCE"},
    {"name": "PVC", "code": "v", "exchange": "DCE"},
    {"name": "聚丙烯", "code": "pp", "exchange": "DCE"},
    {"name": "豆粕", "code": "m", "exchange": "DCE"},
    {"name": "菜粕", "code": "RM", "exchange": "CZCE"},
    {"name": "塑料", "code": "l", "exchange": "DCE"},
]

(train_x, train_y), (val_x, val_y), (test_x, test_y) = util.getData(varieties[0]["code"], 15, None)
