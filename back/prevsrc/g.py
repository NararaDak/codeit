import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lib.QS as qs
import seaborn as sns

titanic = pd.read_csv('D:/01.project/코드잇/part-1-main/data/titanic.csv')
print(titanic.head())
print(titanic.tail())
print(titanic.shape)
qs.Lines()
print(titanic.columns)
qs.Lines()
print(titanic[titanic['Age'].notna()])

qs.Lines()
titanic.info()
qs.Lines()
print(titanic.describe())

qs.Lines()
titanic.sort_values('Age', ascending=True, inplace=True)
print(titanic.head())

qs.Lines()
print(titanic.loc[titanic['PassengerId']==1])

qs.Lines()
print(titanic.loc[titanic['Age']==2,'Age'])

qs.Lines()
print(titanic.iloc[100:110, 3:6])



qs.Lines()
titanic['AgeGroup'] = titanic['Age'] // 10 * 10
print(titanic.tail())
print(titanic[["Age","AgeGroup"]].tail())

print(titanic.head())
print(titanic[["Age","AgeGroup"]].head())

qs.Lines()
titanic.to_excel('D:/01.project/코드잇/part-1-main/data/titanic_with_agegroup.xlsx',index=False)

qs.Lines()
excelData = pd.read_excel('D:/01.project/코드잇/part-1-main/data/titanic_with_agegroup.xlsx')
print(excelData.head())

# sns.histplot(data=titanic, x='Age', bins=20, kde=True)
# plt.show()

# sns.boxplot(data=titanic, x='Age', y='Sex')
# plt.show()

# sns.histplot(titanic["Age"].dropna() , kde=True)
# plt.show()

bike = pd.read_csv('D:/01.project/코드잇/part-1-main/data/bike_sharing.csv')
bike['month'] = pd.to_datetime(bike['datetime']).dt.month
# bike.groupby('month')['count'].mean().plot(kind="bar")
# plt.show()

def MyShow():
    plt.show(block=False)
    plt.pause(3)
    plt.close()

# suvival_rate = titanic.groupby("Sex")["Survived"].mean()
# suvival_rate.plot(kind="bar")
# plt.show()

# sns.boxplot(data=titanic, x='Pclass', y='Fare')
# plt.show()

#Titanic 데이터에서 **성별별 나이 분포**를 박스플롯으로 그려보세요.  
sns.boxplot(data=titanic, x='Sex', y='Age')
MyShow()


#최빈값이 0 이 있는 것으로 봐서 나이가 모르는 사람도 있음.
#최빈값이 0, 24 인데, 중앙값과 평균이 더 높은 것을 보면 나이 많은 사람이 많음.
age = titanic['Age'].dropna()
tmean = age.mean()
tmedian = age.median()
tmode = age.mode()
print(f"평균 : {tmean}\n중앙값: {tmedian}\n최빈값 : {tmode}")

#Bike Sharing 데이터에서 **월별 평균 대여량**을 막대그래프로 시각화하세요.  
bike['month'] = pd.to_datetime(bike['datetime']).dt.month
bike.groupby('month')['count'].mean().plot(kind="bar")
MyShow()


#Bike Sharing 데이터에서 `count` 분포를 히스토그램과 KDE plot으로 시각화하세요. 
sns.histplot(bike["count"].dropna() , bins=30, kde=True)
MyShow()

sns.kdeplot(bike["count"], fill=True)
plt.title("KDE of Bike Demand (count)")
plt.xlabel("count")
plt.ylabel("density")
MyShow()

suvival_rate = titanic.groupby("Pclass")["Survived"].mean()
suvival_rate.plot(kind="bar")
MyShow()
