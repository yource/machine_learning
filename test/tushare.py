'''
Tushare
交易所名称	交易所代码	合约后缀
郑州商品交易所	CZCE	.ZCE
上海期货交易所	SHFE	.SHF
大连商品交易所	DCE	.DCE
中国金融期货交易所	CFFEX	.CFX
上海国际能源交易所	INE	.INE
广州期货交易所	GFEX	.GFE

'''

import test.tushare as ts

ts.set_token('857cf934580eba9e837dae6c05e24696c8f500fc2b153e9ecf9c97c0')
pro = ts.pro_api()
df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')

print(df)