


import pandas as pd
import lib.QS as qs
import matplotlib.pyplot as plt

ages = pd.Series([22,35,58,42])
print(ages)



qs.Lines()
titanic = pd.read_csv('D:/01.project/코드잇/part-1-main/data/titanic.csv')


titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Age'] = titanic['Age'].astype(int)


print(titanic.head())

qs.Lines()
df_left = pd.DataFrame({"ID": [1, 2, 3], "Age": [22, 35, 58]})
df_right = pd.DataFrame({"ID": [2, 3, 4], "City": ["Seoul", "Busan", "Incheon"]})
print(df_left)
print(df_right)


titanic.drop_duplicates(inplace=True)  
print(titanic.shape)


