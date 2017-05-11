# 常量准备
import pandas as pd
from datetime import datetime as dt
from pandas import DataFrame, Series
import time
import datetime

import talib
import numpy as np

#import macd_utils.lib

today = dt.today().strftime('%Y-%m-%d')   # 获得今天的日期
yesterday = datetime.date.today() - datetime.timedelta(days=1)
start = '2017-01-01'                      # starting time
yesterday1=yesterday.strftime('%Y-%m-%d')
max_history_window = 365
end = yesterday1            # ending time
capital_base = 100000
# DataAPI取所有A股
stocks = DataAPI.EquGet(equTypeCD='A',listStatusCD='L',field='secID,nonrestfloatA',pandas="1")
universe = stocks['secID'].tolist()    # 转变为list格式，以便和DataAPI中的格式符合
universe=StockScreener(Factor.LCAP.nsmall(3000))
fileds="secID,turnoverRate,negMarketValue,marketValue,openPrice,closePrice,turnoverVol"
# 条件
# 	前提条件
# 		跌幅＞6个跌停板
# 	必要条件
# 		低位高度密集单峰
# 			10日内，ASR＞80%
# 			不能看见低位筹码密集了，就认为主力建仓了准备拉升了可以买入了，要等待突破；
# 		出击预备
# 			10日内，最高价获利比例＞90%


# 		筹码峰创新高
# 	充分条件
# 		进货模式
# 			确认低位筹码是主力进货
# 		K线走势
# 			慢牛走势
# 			牛长熊短
# 			次低位窄幅横盘
# 			散兵坑
# 		筹码分布
# 			双峰填谷，两次双峰填谷更佳
# 			紫水晶
# 			大盘弱势时放量穿越筹码密集区
# 			90/3
# 		倒向验证
# 			强于大盘
# 			逆势飘红
# 			强势横盘
# 			国投中鲁倒向验证.JPG
# 		其它依据
# 			股东人数减少
def initialize(account):                   # 初始化虚拟账户状态
    account.higt_diefu_rate=0.47
    account.enhance_rate=2
    pass

def handle_data(account):                  # 每个交易日的买入卖出指令
    closePrices=account.get_attribute_history("closePrice", 365)
    #前提条件
    for stock in account.universe:

        #print stock
        flag=tj1(account,closePrices[stock])
        if not flag:
            continue


        #log.info(stock+"======= tj1      ============")
        flag,winner_rate=tj2(account,stock)
        if not flag:
            continue

        #log.info(str(winner_rate)+"=====================tj2 tj2===========================")
        flag=tj3(winner_rate=winner_rate)
        if not  flag:
            continue
        else:
            log.info(stock+"===========  猎豹出击  =================================")


##---------------------------------------      条件区        ---------------------------------------
#满足跌幅的票子
def tj1(account,price_list):
    price=price_list[-1]
    price_list_sort=sorted(price_list)
    high_price=price_list_sort[-1]
    #print str(price)+"=====str(price)==="
    #print str(high_price)+"=====high_price==="
    rate=price/high_price
    if 1-account.higt_diefu_rate>rate:
        return True
    else :
        return False

#满足ASR的票子
def tj2(account,stock):
    current_date=account.current_date
    keys,values,closePrice,winner_rate=get_stcok_chouma_status(universe,current_date,account,stock)
    rate=priceASR(closePrice,keys,values)
    flag=isRightASR(rate)
    return flag,winner_rate

#满足获利比例的票子
def tj3(winner_rate):
    if winner_rate>0.9:
        return True
    else:
        return False

##---------------------------------------      方法区        ---------------------------------------
def countCYQ(data,account):
    turnoverRates=data['turnoverRate'] #换手率
    closePrices=data['closePrice'] #收盘价
    openPrices=data['openPrice'] #开盘价
    turnoverVols=data['turnoverVol'] #成交量

    map=getPrice_Vol_map(turnoverRates,closePrices,openPrices,turnoverVols,account)

    dataframe_len=len(closePrices)
    closePrice=closePrices[dataframe_len-1]

    #画图
    keys=map.keys()

    keys=sorted(keys)

    i=0
    values=[]

    while i<len(keys):
        key=keys[i]
        value=map[key]
        values.append(value)
        i+=1

    #获利比例
    winner_rate=winner(keys,values,closePrice)
    return keys,values,closePrice,winner_rate




def getDay(gap_num,date):
    date = date - datetime.timedelta(days=gap_num)
    new_date=date.strftime('%Y-%m-%d')
    return new_date

def getDay2(gap_num):
    yesterday = datetime.date.today() - datetime.timedelta(days=gap_num)
    yesterday1=yesterday.strftime('%Y-%m-%d')
    return yesterday1

#得到今日成本
def getPreAveragePrice(pre_price,today_turnoverRate,closePrice):
    today_price=(1-today_turnoverRate)*pre_price+today_turnoverRate*closePrice
    return today_price



def getPrice_Vol_map(turnoverRates,closePrices,openPrices,turnoverVols,account):
    pre_cost=0.00
    cost_vol_map={}

    num=len(turnoverRates)
    for index,var in enumerate(turnoverRates):
        price=(closePrices[index]+openPrices[index])/2
        pre_cost=price

        #rest_vol=(0.8/(index+1))*turnoverVols[index]*turnoverRates[index]
        chouma=get_chouma(index,turnoverVols[index],turnoverRates,account)
        #rest_vol=(-0.16*(index+1)+0.96)*turnoverVols[index]*turnoverRates[index]
        pre_cost=round(pre_cost,2)

        if cost_vol_map.has_key(pre_cost):
            pre_chouma=cost_vol_map.get(pre_cost)
            chouma=chouma+pre_chouma
            chouma=int(chouma)
            cost_vol_map[pre_cost]=chouma

        chouma=int(chouma)
        cost_vol_map[pre_cost]=chouma
    return cost_vol_map


#累计获利筹码
def get_chouma(my_index,turnoverVol,turnoverRates,account):
    chouma=0
    total=len(turnoverRates)
    #print str(my_index)+"============my_index================"

    turnoverRates=turnoverRates[my_index:total]
    total_2=len(turnoverRates)
    for index,var in enumerate(turnoverRates):
        if index>=total_2-1:
            pass
        else:
            rate=1-var*account.enhance_rate
            #print rate
            turnoverVol=turnoverVol*(rate)
            #print rate
            #print turnoverVol

    return turnoverVol

def winner(keys,values,price):
    winner_rate=0.00
    total=sum(values)
    winner_vol=0.00
    for index,var in enumerate(keys):
        if price >=var:
            winner_vol+=values[index]


    winner_rate=winner_vol/total
    return winner_rate

def get_stcok_chouma_status(universe,end,account,stock):
    #print account
    #print(account.current_date)
    trade_day=getDay(1,end)
    re_day=getDay(1000,end)
    data=DataAPI.MktEqudGet(tradeDate=u"",secID=stock,ticker=u"",beginDate=re_day,endDate=trade_day,isOpen="1",field=fileds,pandas="1")
    keys,values,closePrice,winner_rate=countCYQ(data,account)
    return keys,values,closePrice,winner_rate

#返回rate
def priceASR(price,keys,values):
    point_1=price*(1+0.1)
    point_2=price*(1-0.1)

    part_vol=0.00
    total_vol=sum(values)
    for index ,var in enumerate(keys) :
        if var <= point_1 and var >=point_2:
            part_vol+=values[index]

    rate=part_vol/total_vol;
    return  rate

def isRightASR(rate):
    if rate>0.9:
        return True
    else:
        return False
