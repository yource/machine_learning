from datetime import datetime, date
from contextlib import closing
from tqsdk import TqApi, TqAuth, TqSim
from tqsdk.tools import DataDownloader

contracts = [
    {
        "name": "甲醇",
        "code": "MA",
        "exchange": "CZCE",
    },
    {
        "name": "PVC",
        "code": "v",
        "exchange": "DCE",
    },
    {
        "name": "聚丙烯",
        "code": "pp",
        "exchange": "DCE",
    },
    {
        "name": "豆粕",
        "code": "m",
        "exchange": "DCE",
    },
    {
        "name": "菜粕",
        "code": "RM",
        "exchange": "CZCE",
    },
    {
        "name": "塑料",
        "code": "l",
        "exchange": "DCE",
    },
    {"name": "尿素", "code": "UR", "exchange": "CZCE"},
    {"name": "锰硅", "code": "SM", "exchange": "CZCE"},
    {"name": "鸡蛋", "code": "jd", "exchange": "DCE"},
    {"name": "乙二醇", "code": "eg", "exchange": "DCE"},
    {"name": "螺纹钢", "code": "rb", "exchange": "SHFE"},
    {"name": "热卷", "code": "hc", "exchange": "SHFE"},
    {"name": "燃油", "code": "fu", "exchange": "SHFE"},
    {"name": "纯碱", "code": "SA", "exchange": "CZCE"},
    {"name": "玻璃", "code": "FG", "exchange": "CZCE"},
]
contractList = []
for m, item in enumerate(contracts):
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 5 * 60,
            "filename": "data/main/" + item["code"] + "_5min.csv",
        }
    )
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 10 * 60,
            "filename": "data/main/" + item["code"] + "_10min.csv",
        }
    )
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 15 * 60,
            "filename": "data/main/" + item["code"] + "_15min.csv",
        }
    )
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 20 * 60,
            "filename": "data/main/" + item["code"] + "_20min.csv",
        }
    )
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 30 * 60,
            "filename": "data/main/" + item["code"] + "_30min.csv",
        }
    )
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 60 * 60,
            "filename": "data/main/" + item["code"] + "_60min.csv",
        }
    )
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 90 * 60,
            "filename": "data/main/" + item["code"] + "_90min.csv",
        }
    )
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 120 * 60,
            "filename": "data/main/" + item["code"] + "_120min.csv",
        }
    )
    contractList.append(
        {
            "exchange": item["exchange"],
            "name": item["name"],
            "code": item["code"],
            "sec": 24 * 60 * 60,
            "filename": "data/main/" + item["code"] + "_24hour.csv",
        }
    )


# print("contractList",contractList)

api = TqApi(auth=TqAuth("15150685244", "yangxiang88"))

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

maxIdx = len(contractList) - 1
curIdx = 0


def getDownloader(idx):
    task = None
    ct = contractList[idx]
    try:
        task = DataDownloader(
            api,
            dur_sec=ct["sec"],
            symbol_list="KQ.m@"+ct["exchange"] + "." + ct["code"],
            start_dt=date(2016, 1, 1),
            end_dt=date(2023, 6, 25),
            csv_file_name=ct["filename"],
        )
        return task
    except Exception as e:
        print("合约获取出错" + contractList[curIdx]["name"])
        return getDownloader(idx + 1)


download_task = getDownloader(0)
currentErr = ""
with closing(api):
    while curIdx <= maxIdx:
        try:
            api.wait_update()
        except Exception as e:
            print("合约获取出错" + contractList[curIdx]["name"], e)
            currentErr = contractList[curIdx]["code"]

        if download_task.is_finished():
            curIdx += 1
            if curIdx <= maxIdx:
                download_task = getDownloader(curIdx)
                print("开始下载 " + contractList[curIdx]["name"])
            else:
                print("下载完成～")
        else:
            print("[" + str(curIdx + 1) + "/" + str(maxIdx + 1) + "]" + contractList[curIdx]["name"] + ": ", download_task.get_progress())

print("=== 下载进程结束 ===")
