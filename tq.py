'''
QQ请联系532428198 或电话联系:4008207951 转 9
日盘 09:00~10:15 10:30:11:30 13:30:15:00
夜盘 21:00~23:00+
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

期货品种分类：
1. 金属板块：
  1.1 贵金属：黄金 白银 
  1.2 有色金属：铜 锌 铝 锡 铅 镍 

2. 黑色板块：铁矿石 螺纹钢 热卷 不锈钢 焦炭 焦煤 硅铁 锰硅 动力煤

3. 能源化工：
  3.1 能源板块：原油 燃油 液化石油气 沥青
  3.2 化工板块：聚乙烯 聚丙烯 PVC 乙二醇 苯乙烯 PTA 甲醇 玻璃 橡胶 纸浆 

4. 农产品：
  4.1 大豆家族：豆一 豆二 豆油 豆粕
  4.2 玉米家族：玉米 玉米淀粉
  4.3 菜籽家族：菜籽 菜粕 菜油
  4.4 棉花家族：棉花 棉纱
  4.5 水稻家族：早稻 晚稻 粳稻 
  4.6 吉祥如意：苹果 红枣 花生 白糖
  4.7 家禽家族：鸡蛋 生猪
  ----
  4.8 油脂板块：棕榈油 豆油 菜油 
  4.9 谷物饲料：玉米 豆一 豆二 豆粕 菜粕 
  4.10 软商品农副：白糖 棉花 鸡蛋 苹果 

'''


from tqsdk import TqApi, TqAuth


api = TqApi(auth=TqAuth("18655533530", "YangXiang88"))
quote = api.get_quote("DCE.b2305")

while True:
    print("等待...")
    api.wait_update()
    print (quote.datetime, quote.last_price)