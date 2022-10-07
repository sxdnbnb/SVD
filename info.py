# _*_coding : utf-8 _*_
# @Time : 2022/10/4 15:14
# @Author : SunShine
# @File : info
# @Project : SVD
import pandas as pd

data = pd.read_csv("u4.test", delim_whitespace=True,  # 或者 sep='\t',
                   header=None, names=['user id', 'item id', 'rating', "timestamp"])
print(data.head())
# 分数平均值
av=data["rating"].mean()
print(av)
print(data["user id"].max())
print(data["user id"].count())
