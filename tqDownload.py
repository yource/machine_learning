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

types = [
    {
        "name": "黑色金属", #板块，作文件夹名
        "list": [
            {
                "name":"铁矿石", 
                "exchange":"DCE", #交易所
                "code":"i", #代码
                "main": ["01","05","09"],
                "list": ['1405',]
            }
        ]
    }
]

# 每个月都是主力: 不锈钢/沪铜
contracts = [
    {
        "type":"黑色金属", "name":"铁矿石", "code":"i", "exchange":"DCE", 
        "list":["1605","1609",
                "1701","1705","1709","1801","1805","1809","1901","1905","1909",
                "2001","2005","2009","2101","2105","2109","2201","2205","2209",
                "2301","2305"]
    },{
        "type":"黑色金属", "name":"螺纹钢", "code":"rb", "exchange":"SHFE", 
        "list":["1605","1610",
                "1701","1705","1710","1801","1805","1810","1901","1905","1910",
                "2001","2005","2010","2101","2105","2110","2201","2205","2210",
                "2301","2305"]
    },{
        "type":"黑色金属", "name":"热卷", "code":"hc", "exchange":"SHFE", 
        "list":["1605","1610",
                "1701","1705","1710","1801","1805","1810","1901","1905","1910",
                "2001","2005","2010","2101","2105","2110","2201","2205","2210",
                "2301","2305"]
    },{
        "type":"有色金属", "name":"沪铜", "code":"cu", "exchange":"SHFE", 
        "list":["1605","1610",
                "1701","1705","1710","1801","1805","1810","1901","1905","1910",
                "2001","2005","2010","2101","2105","2110","2201","2205","2210",
                "2301","2305"]
    }
]

contractMain = [
    {}
]

api = TqApi(auth=TqAuth("18655533530", "YangXiang88"))
download_tasks = {}
download_tasks["rb1805"] = DataDownloader(api, symbol_list="SHFE.rb1805", dur_sec=24*60*60,
                    start_dt=date(2018, 1, 1), end_dt=date(2018, 9, 1), csv_file_name="./data/test/rb1805.csv")

with closing(api):
    while not all([v.is_finished() for v in download_tasks.values()]):
        api.wait_update()
        print("progress: ", { k:("%.2f%%" % v.get_progress()) for k,v in download_tasks.items() })