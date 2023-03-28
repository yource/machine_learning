'''
SHFE 上期所 上海期货交易所
    沪金au 沪银ag 沪铝au 沪铜cu 沪镍ni 沪铅pb 沪锡sn  沪锌zn
    沥青bu 燃油fu 热卷hc 橡胶ru 纸浆sp 螺纹钢rb 不锈钢ss 线材wr

DCE 大商所 大连商品交易所
    豆一a 豆二b 玉米c 淀粉cs 鸡蛋jd 生猪lh 豆粕m 棕榈油p 粳米rr 豆油y
    胶合板bb 纤维板fb 塑料l 苯乙烯eb 乙二醇eg LPG(液化石油气)pg 聚丙烯pp PVC(聚氯乙烯)v 
    铁矿石i 焦炭j 焦煤jm 
    
CZCE 郑商所 郑州商品交易所
    苹果AP 红枣CJ 粳稻JR 晚籼稻LR 菜油OI 花生PK 普麦PM 早籼稻RI 菜粕RM 菜籽RS 强麦WH 
    甲醇MA 纯碱SA 硅铁SF 锰硅SM  PTA(精对苯二甲酸)TA 尿素UR
    棉纱CY 棉花CF 玻璃FG 短纤PF 白糖SR 动力煤ZC 

INE 上期能源 上海能源中心
    国际铜bc 低硫燃油lu 20号胶nr 原油sc
'''

from datetime import datetime, date
from contextlib import closing
from tqsdk import TqApi, TqAuth, TqSim
from tqsdk.tools import DataDownloader

api = TqApi(auth=TqAuth("18655533530", "YangXiang88"))

download_tasks = {}
# 下载从 2018-01-01 到 2018-09-01 的 SR901 日线数据
download_tasks["SR_daily"] = DataDownloader(api, symbol_list="CZCE.SR901", dur_sec=24*60*60,
                    start_dt=date(2018, 1, 1), end_dt=date(2018, 9, 1), csv_file_name="SR901_daily.csv")