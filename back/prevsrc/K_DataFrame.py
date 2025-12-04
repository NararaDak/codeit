import pandas as pd
import lib.QS as qs
import matplotlib.pyplot as plt

ages = pd.Series([22,35,58,42])
print(ages)



qs.Lines()
titanic = pd.read_csv('D:/01.project/코드잇/part-1-main/data/titanic.csv')
titanic.set_index("Name")
print(titanic.head())


qs.Lines()
titanic_drop = titanic.dropna(subset=["Age"])
print(titanic.shape)
qs.Lines()
print(titanic_drop.shape)



mean_age=titanic["Age"].mean()

titanic["Age_filled"] = titanic["Age"].fillna(mean_age)

names = pd.Series([" Alice ", "BOB", "cHaRlIe "])
print(names.str.lower())


mylam =  lambda x : x.split(",")[0]

qs.Lines()
# titanic["Name_short"] = titanic["Name"].str.split(",").str[0]
titanic = titanic.assign(Name_short=lambda df: df["Name"].apply(lambda s: s.split(",")[0]))
print(titanic.head())


# 이름에서 호칭 추출
titanic['Initial']=0
for i in titanic:
    titanic['Initial']=titanic.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
print(titanic.head())

titanic['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
qs.Lines()
print(titanic.groupby('Initial')['Age'].mean() )#lets check the average age by Initials

qs.Lines()
print(titanic.head())

s = titanic.groupby('Initial')['Survived'].sum().sort_values(ascending=False)
qs.Lines()
print(s)
plt.bar(s.index, s.values)  # x: 호칭(5개), y: 생존자 수(5개)
plt.xlabel('Initial')
plt.ylabel('Survived Sum')
plt.show()

plt.plot(s.index, s.values, marker='o', linestyle='-')  # 점 + 연결선
plt.xlabel('Initial'); plt.ylabel('Survived Sum')
plt.grid(True, alpha=0.3)
plt.show()

s = titanic.groupby('Initial')['Survived'].sum().sort_values(ascending=False)
fig, ax = plt.subplots()
bars = ax.bar(s.index, s.values)

# 막대 위 가운데에 y값 표기
ax.bar_label(bars, labels=[f'{int(v)}' for v in s.values], padding=3)
ax.set_xlabel('Initial'); ax.set_ylabel('Survived Sum')
plt.show()

