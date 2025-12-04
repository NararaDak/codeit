import pandas as pd
import lib.QS as qs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')

qs.MyClear()


qs.Lines()
titanic = pd.read_csv('D:/01.project/코드잇/part-1-main/data/titanic.csv')
print(titanic.head())


print(titanic["Age"].isna().sum())
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())
print(titanic["Age"])


Q1 = titanic["Fare"].quantile(0.25)
Q3 = titanic["Fare"].quantile(0.75)
IQR = Q3 - Q1 
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

qs.Lines()
print(f"lower:{lower_bound},upper:{upper_bound}")

outliers = titanic[(titanic["Fare"] < lower_bound )| (upper_bound < titanic["Fare"])]
qs.Lines()
print(outliers.head())
qs.Lines()
print(outliers["Fare"].head())

