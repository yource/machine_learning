from datetime import datetime, date
from contextlib import closing
from tqsdk import TqApi, TqAuth, TqSim
from tqsdk.tools import DataDownloader

contracts = [
    # {
    #     "type":"农产品", "name":"鸡蛋", "code":"jd", "exchange":"DCE", 
    #     "list":["1603","1605","1609","1611",
    #             "1701","1703","1705","1707","1709","1711",
    #             "1801","1803","1805","1807","1809","1811",
    #             "1901","1903","1905","1907","1909","1911",
    #             "2001","2003","2005","2007","2009","2011",
    #             "2101","2103","2105","2107","2109","2111",
    #             "2201","2203","2205","2207","2209","2211",
    #             "2301","2303","2305","2307","2309"]
    # },
    # {
    #     "type":"农产品", "name":"玉米", "code":"c", "exchange":"DCE",
    #     "list":["1603","1605","1607","1609","1611",
    #             "1701","1703","1705","1707","1709","1711",
    #             "1801","1803","1805","1807","1809","1811",
    #             "1901","1903","1905","1907","1909","1911",
    #             "2001","2003","2005","2007","2009","2011",
    #             "2101","2103","2105","2107","2109","2111",
    #             "2201","2203","2205","2207","2209","2211",
    #             "2301","2303","2305","2307","2309"]
    # },
    # {
    #     "type":"农产品", "name":"淀粉", "code":"cs", "exchange":"DCE",
    #     "list":["1603","1605","1607","1609","1611",
    #             "1701","1703","1705","1707","1709","1711",
    #             "1801","1803","1805","1807","1809","1811",
    #             "1901","1903","1905","1907","1909","1911",
    #             "2001","2003","2005","2007","2009","2011",
    #             "2101","2103","2105","2107","2109","2111",
    #             "2201","2203","2205","2207","2209","2211",
    #             "2301","2303","2305","2307","2309"]
    # },
    # { 
    #     "type":"黑色金属", "name":"热卷", "code":"hc", "exchange":"SHFE",
    #     "list":["1603","1604","1605","1606","1607","1608","1609","1610","1611","1612",
    #             "1701","1702","1703","1704","1705","1706","1707","1708","1709","1710","1711","1712",
    #             "1801","1802","1803","1804","1805","1806","1807","1808","1809","1810","1811","1812",
    #             "1901","1902","1903","1904","1905","1906","1907","1908","1909","1910","1911","1912",
    #             "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012",
    #             "2101","2102","2103","2104","2105","2106","2107","2108","2109","2110","2111","2112",
    #             "2201","2202","2203","2204","2205","2206","2207","2208","2209","2210","2211","2212",
    #             "2301","2302","2303","2304","2305","2306","2307","2308","2309"]
    # },
    # {
    #     "type":"黑色金属", "name":"螺纹钢", "code":"rb", "exchange":"SHFE",
    #     "list":["1603","1604","1605","1606","1607","1608","1609","1610","1611","1612",
    #             "1701","1702","1703","1704","1705","1706","1707","1708","1709","1710","1711","1712",
    #             "1801","1802","1803","1804","1805","1806","1807","1808","1809","1810","1811","1812",
    #             "1901","1902","1903","1904","1905","1906","1907","1908","1909","1910","1911","1912",
    #             "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012",
    #             "2101","2102","2103","2104","2105","2106","2107","2108","2109","2110","2111","2112",
    #             "2201","2202","2203","2204","2205","2206","2207","2208","2209","2210","2211","2212",
    #             "2301","2302","2303","2304","2305","2306","2307","2308","2309"]
    # },
    # {
    #     "type":"煤炭板块", "name":"焦煤", "code":"jm", "exchange":"DCE",
    #     "list":["1603","1604",
    # {
    #     "type":"煤炭板块", "name":"焦煤", "code":"jm", "exchange":"DCE",
    #     "list":["1605","1606","1607","1608","1609","1610","1611","1612",
    #             "1701","1702","1703","1704","1705","1706","1707","1708","1709","1710","1711","1712",
    #             "1801","1802","1803","1804","1805","1806","1807","1808","1809","1810","1811","1812",
    #             "1901","1902","1903","1904","1905","1906","1907","1908","1909","1910","1911","1912",
    #             "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012",
    #             "2101","2102","2103","2104","2105","2106","2107","2108","2109","2110","2111","2112",
    #             "2201","2202","2203","2204","2205","2206","2207","2208","2209","2210","2211","2212",
    #             "2301","2302","2303","2304","2305","2306","2307","2308","2309"]
    # },
    # {
    #     "type":"煤炭板块", "name":"焦炭", "code":"j", "exchange":"DCE",
    #     "list":["1603","1604","1605","1606","1607","1608","1609","1610","1611","1612",
    #             "1701","1702","1703","1704","1705","1706","1707","1708","1709","1710","1711","1712",
    #             "1801","1802","1803","1804","1805","1806","1807","1808","1809","1810","1811","1812",
    #             "1901","1902","1903","1904","1905","1906","1907","1908","1909","1910","1911","1912",
    #             "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012",
    #             "2101","2102","2103","2104","2105","2106","2107","2108","2109","2110","2111","2112",
    #             "2201","2202","2203","2204","2205","2206","2207","2208","2209","2210","2211","2212",
    #             "2301","2302","2303","2304","2305","2306","2307","2308","2309"]
    # },{
    #     "type":"黑色金属", "name":"铁矿石", "code":"i", "exchange":"DCE",
    #     "list":["1603","1604","1605","1606","1607","1608","1609","1610","1611","1612",
    #             "1701","1702","1703","1704","1705","1706","1707","1708","1709","1710","1711","1712",
    #             "1801","1802","1803","1804","1805","1806","1807","1808","1809","1810","1811","1812",
    #             "1901","1902","1903","1904","1905","1906","1907","1908","1909","1910","1911","1912",
    #             "2001","2002","2003","2004","2005","2006","2007","2008","2009","2010","2011","2012",
    #             "2101","2102","2103","2104","2105","2106","2107","2108","2109","2110","2111","2112",
    #             "2201","2202","2203","2204","2205","2206","2207","2208","2209","2210","2211","2212",
    #             "2301","2302","2303","2304","2305","2306","2307","2308","2309"]
    # },{
    #     "type":"油脂油料", "name":"豆一", "code":"a", "exchange":"DCE", 
    #     "list":["1603","1605","1607","1609","1611",
    #             "1701","1703","1705","1707","1709","1711",
    #             "1801","1803","1805","1807","1809","1811",
    #             "1901","1903","1905","1907","1909","1911",
    #             "2001","2003","2005","2007","2009","2011",
    #             "2101","2103","2105","2107","2109","2111",
    #             "2201","2203","2205","2207","2209","2211",
    #             "2301","2303","2305","2307","2309"]
    # },
    {
        "type":"油脂油料", "name":"豆二", "code":"b", "exchange":"DCE", 
        "list":["1605","1607","1609","1611",
                "1701","1703","1705","1707","1709","1711",
                "1801","1803","1805","1807","1809","1811",
                "1901","1903","1905","1907","1909","1911",
                "2001","2003","2005","2007","2009","2011",
                "2101","2103","2105","2107","2109","2111",
                "2201","2203","2205","2207","2209","2211",
                "2301","2303","2305","2307","2309"]
    }
]

contractList = []
for m,item in enumerate(contracts):
    for n,subItem in enumerate(item["list"]):
        contractList.append({
            "exchange": item["exchange"],
            "name": item["name"]+subItem,
            "code": item["code"]+subItem,
            "sec": 5*60,
            "filename": "./data/contracts/contract5min/"+item["code"]+"/"+item["code"]+subItem+"_5min.csv"
        })
        contractList.append({
            "exchange": item["exchange"],
            "name": item["name"]+subItem,
            "code": item["code"]+subItem,
            "sec": 30*60,
            "filename": "./data/contracts/contract30min/"+item["code"]+"/"+item["code"]+subItem+"_30min.csv"
        })
        contractList.append({
            "exchange": item["exchange"],
            "name": item["name"]+subItem,
            "code": item["code"]+subItem,
            "sec": 60*60,
            "filename": "./data/contracts/contract60min/"+item["code"]+"/"+item["code"]+subItem+"_60min.csv"
        })
        contractList.append({
            "exchange": item["exchange"],
            "name": item["name"]+subItem,
            "code": item["code"]+subItem,
            "sec": 24*60*60,
            "filename": "./data/contracts/contractDaily/"+item["code"]+"/"+item["code"]+subItem+"_daily.csv"
        })

# print("contractList",contractList)

# 主连
contractMain = [
    { "type":"黑色金属", "name":"铁矿石", "code":"i", "exchange":"DCE" },
    { "type":"黑色金属", "name":"螺纹钢", "code":"rb", "exchange":"SHFE" },
    { "type":"黑色金属", "name":"热卷", "code":"hc", "exchange":"SHFE" },
    { "type":"黑色金属", "name":"不锈钢", "code":"ss", "exchange":"SHFE" },
    { "type":"黑色金属", "name":"硅铁", "code":"SF", "exchange":"CZCE" },
    { "type":"黑色金属", "name":"锰硅", "code":"SM", "exchange":"CZCE" },
    { "type":"有色金属", "name":"沪铜", "code":"cu", "exchange":"SHFE" },
    { "type":"有色金属", "name":"沪铝", "code":"al", "exchange":"SHFE" },
    { "type":"有色金属", "name":"沪锌", "code":"zn", "exchange":"SHFE" },
    { "type":"有色金属", "name":"沪铅", "code":"pb", "exchange":"SHFE" },
    { "type":"有色金属", "name":"沪镍", "code":"ni", "exchange":"SHFE" },
    { "type":"有色金属", "name":"沪锡", "code":"sn", "exchange":"SHFE" },
    { "type":"有色金属", "name":"国际铜", "code":"bc", "exchange":"INE" },
    { "type":"有色金属", "name":"工业硅", "code":"si", "exchange":"GFEX" },
    { "type":"贵金属", "name":"沪金", "code":"au", "exchange":"SHFE" },
    { "type":"贵金属", "name":"沪银", "code":"ag", "exchange":"SHFE" },
    { "type":"油脂油料", "name":"豆一", "code":"a", "exchange":"DCE" },
    { "type":"油脂油料", "name":"豆二", "code":"b", "exchange":"DCE" },
    { "type":"油脂油料", "name":"豆油", "code":"y", "exchange":"DCE" },
    { "type":"油脂油料", "name":"豆粕", "code":"m", "exchange":"DCE" },
    { "type":"油脂油料", "name":"棕榈油", "code":"p", "exchange":"DCE" },
    { "type":"油脂油料", "name":"菜油", "code":"OI", "exchange":"CZCE" },
    { "type":"油脂油料", "name":"菜粕", "code":"RM", "exchange":"CZCE" },
    { "type":"农产品", "name":"玉米", "code":"c", "exchange":"DCE" },
    { "type":"农产品", "name":"淀粉", "code":"cs", "exchange":"DCE" },
    { "type":"农产品", "name":"鸡蛋", "code":"jd", "exchange":"DCE" },
    { "type":"农产品", "name":"棉花", "code":"CF", "exchange":"CZCE" },
    { "type":"农产品", "name":"面纱", "code":"CY", "exchange":"CZCE" },
    { "type":"农产品", "name":"苹果", "code":"AP", "exchange":"CZCE" },
    { "type":"农产品", "name":"红枣", "code":"CJ", "exchange":"CZCE" },
    { "type":"农产品", "name":"花生", "code":"PK", "exchange":"CZCE" },
    { "type":"农产品", "name":"粳米", "code":"rr", "exchange":"DCE" },
    { "type":"农产品", "name":"生猪", "code":"lh", "exchange":"DCE" },
    { "type":"农产品", "name":"白糖", "code":"SR", "exchange":"CZCE" },
    { "type":"能源化工", "name":"原油", "code":"sc", "exchange":"INE" },
    { "type":"能源化工", "name":"LPG", "code":"pg", "exchange":"DCE" },
    { "type":"能源化工", "name":"低硫燃油", "code":"lu", "exchange":"INE" },
    { "type":"能源化工", "name":"沥青", "code":"bu", "exchange":"SHFE" },
    { "type":"能源化工", "name":"甲醇", "code":"MA", "exchange":"CZCE" },
    { "type":"能源化工", "name":"乙二醇", "code":"eg", "exchange":"DCE" },
    { "type":"能源化工", "name":"塑料", "code":"l", "exchange":"DCE" },
    { "type":"能源化工", "name":"PTA", "code":"TA", "exchange":"CZCE" },
    { "type":"能源化工", "name":"聚丙烯", "code":"pp", "exchange":"DCE" },
    { "type":"能源化工", "name":"苯乙烯", "code":"eb", "exchange":"DCE" },
    { "type":"能源化工", "name":"尿素", "code":"UR", "exchange":"CZCE" },
    { "type":"能源化工", "name":"橡胶", "code":"ru", "exchange":"SHFE" },
    { "type":"能源化工", "name":"20号胶", "code":"nr", "exchange":"INE" },
    { "type":"能源化工", "name":"纸浆", "code":"sp", "exchange":"SHFE" },
    { "type":"能源化工", "name":"短纤", "code":"PF", "exchange":"CZCE" },
    { "type":"能源化工", "name":"燃油", "code":"fu", "exchange":"SHFE" },
    { "type":"能源化工", "name":"PVC", "code":"v", "exchange":"DCE" },
    { "type":"能源化工", "name":"纯碱", "code":"SA", "exchange":"CZCE" },
    { "type":"能源化工", "name":"玻璃", "code":"FG", "exchange":"CZCE" },
    { "type":"煤炭板块", "name":"焦煤", "code":"jm", "exchange":"DCE" },
    { "type":"煤炭板块", "name":"焦炭", "code":"j", "exchange":"DCE" },
]

api = TqApi(auth=TqAuth("18655533530", "YangXiang88"))

# # 下载示例
# download_tasks={}
# # # 下载从 2018-01-01 到 2018-09-01 的 SR901 日线数据
# download_tasks["SR_daily"] = DataDownloader(api, symbol_list="CZCE.SR901", dur_sec=24*60*60,
#                     start_dt=date(2016, 1, 1), end_dt=date(2018, 9, 1), csv_file_name="SR901_daily.csv")
# # # 下载从 2017-01-01 到 2018-09-01 的 rb主连 5分钟线数据
# download_tasks["j1901_5min"] = DataDownloader(api, symbol_list="DCE.j1901", dur_sec=5*60,
#                     start_dt=date(2016, 1, 1),end_dt=date(2023, 3, 29),  csv_file_name="./data/test/j1901_5min.csv")
# with closing(api):
#     while not all([v.is_finished() for v in download_tasks.values()]):
#         api.wait_update()
#         print("progress: ", { k:("%.2f%%" % v.get_progress()) for k,v in download_tasks.items() })

maxIdx = len(contractList)-1
curIdx = 0

def getDownloader(idx):
    task = None
    ct = contractList[idx]
    try: 
        task = DataDownloader(api, dur_sec = ct["sec"],
                          symbol_list = ct["exchange"]+"."+ct["code"],
                          start_dt = date(2016, 1, 1), end_dt = date(2023, 4, 10), 
                          csv_file_name = ct["filename"])
        return task
    except Exception as e:
        print("合约获取出错"+contractList[curIdx]["name"])
        return getDownloader(idx+1)

download_task= getDownloader(0)
currentErr = ""
with closing(api):
    while (curIdx<=maxIdx):
        try:
            api.wait_update()
        except Exception as e:
            print("合约获取出错"+contractList[curIdx]["name"],e)
            currentErr = contractList[curIdx]["code"]
        
        if download_task.is_finished():
            curIdx += 1
            if curIdx<=maxIdx:
                download_task= getDownloader(curIdx)
                print("开始下载 "+contractList[curIdx]["name"])
            else:
                print("下载完成～")
        else:
            print("["+str(curIdx+1)+"/"+str(maxIdx+1)+"]"+contractList[curIdx]["name"]+": ", download_task.get_progress())
        
print("=== 下载进程结束 ===")
