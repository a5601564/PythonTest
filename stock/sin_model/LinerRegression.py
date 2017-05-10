from datetime import time

import pandas as pd
from io import StringIO

from sklearn import linear_model

import matplotlib.pyplot as plt



# 房屋面积与价格历史数据(csv文件)
csv_data = 'square_feet,price\n150,6450\n200,7450\n250,8450\n300,9450\n350,11450\n400,15450\n600,18450\n'

# 读入dataframe
df = pd.read_csv(StringIO(csv_data))
print(df)


# 建立线性回归模型
regr = linear_model.LinearRegression()

# 拟合
regr.fit(df['square_feet'].reshape(-1, 1), df['price']) # 注意此处.reshape(-1, 1)，因为X是一维的！

# 不难得到直线的斜率、截距
a, b = regr.coef_, regr.intercept_

# 给出待预测面积
area = 238.5

# 方式1：根据直线方程计算的价格
print(a * area + b)

# 方式2：根据predict方法预测的价格
print(regr.predict(area))
date = time.strptime(str('2017-05-06'), "%Y-%m-%d")
# 画图
# 1.真实的点
plt.scatter(df['square_feet'], df['price'], color='blue')

# 2.拟合的直线
plt.plot(df['square_feet'], regr.predict(df['square_feet'].reshape(-1,1)), color='red', linewidth=4)

plt.show()

#返回rate
def priceASR(price,keys,values):
    point_1=price*(1+0.1)
    point_2=price*(1-0.1)

    part_vol=0.00
    total_vol=sum(values)
    for index ,var in enumerate(keys) :
        if var <= point_1 and var >=point_2:
            part_vol+=values(index)

    rate=part_vol/total_vol;
    return  rate

def isRightASR(rate):
    if rate>0.9:
        return True
    else: 
        return False



