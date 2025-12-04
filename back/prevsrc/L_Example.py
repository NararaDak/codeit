import pandas as pd
import lib.QS as qs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')




qs.Lines()
titanic = pd.read_csv('D:/01.project/코드잇/part-1-main/data/titanic.csv')
print(titanic.head())

# # seaborn 이용.
# # 성별과 객실 등급별 생존률 막대 차트
# pd.crosstab([titanic.Sex,titanic.Survived],titanic.Pclass,margins=True).style.background_gradient(cmap='summer_r')
# # seaborn 이 Y 값을 자동으로 평균으로 한다.
# sns.catplot(x='Pclass', y='Survived', hue='Sex', data=titanic, kind='point')
# # qs.MyShow(plt)
# plt.xlabel("객실등급")
# plt.ylabel("생존율")
# qs.MyShow(plt)

# sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic)
# qs.MyShow(plt)

# 문제 1.
qs.Lines()
survive_ratio = titanic.groupby("Sex")["Survived"].mean().reset_index()
survive_ratio.columns=["sex","survival_rate"]
print(survive_ratio)

# 문제 2.
qs.Lines()
mean_age_byClass = titanic.groupby("Pclass")["Age"].mean().reset_index()
mean_age_byClass.columns=["pclass","mean_age"] 
print(mean_age_byClass)





bike = pd.read_csv('D:/01.project/코드잇/part-1-main/data/bike_sharing.csv')
qs.Lines()

#문제 3.
# 인덱스를 지정 한다. 
bike["datetime"] = pd.to_datetime(bike["datetime"])
bike.set_index("datetime",inplace=True)
# weekday 컬럼 추가. 
bike['weekday'] = bike.index.weekday
avg_cnt = bike.groupby("weekday")["count"].mean().reset_index()
avg_cnt.columns = ["weekday","count"]
avg_cnt = avg_cnt.sort_values("count")  # count 컬럼 오름차순 정렬
print(avg_cnt)

#문제 4.

count_by_weather = bike.groupby("weather")["count"].mean().reset_index()
count_by_weather.columns=["weather","avg_count"]  # count 대신 avg_count로 변경

# Weather 값을 한글로 매핑
weather_mapping = {1: "맑음", 2: "흐림", 3: "비/눈", 4: "심각"}
count_by_weather["weather_korean"] = count_by_weather["weather"].map(weather_mapping)

qs.Lines()
print(count_by_weather)

# 그래프 크기 설정
plt.figure(figsize=(10, 6))

plt.bar(count_by_weather.weather_korean, count_by_weather.avg_count)  
plt.xlabel("날씨")  
plt.ylabel("평균 대여 수")
plt.title("날씨별 평균 대여 수")
plt.tight_layout()  # 레이아웃 자동 조정
qs.MyShow(plt)


#문제 5.
# Age 컬럼의 결측값 처리
titanic["Age"] = titanic["Age"].dropna()
titanic["agegroup"] = titanic["Age"].apply(lambda x: "Adults" if x > 18 else "Children")
#titanic["agegroup"] = titanic["Age"].apply(lambda x: "Child" if x <= 18 else "Adult")
survive_ratio = titanic.groupby(["agegroup", "Sex"])["Survived"].mean().reset_index()
survive_ratio["Survived"] = survive_ratio["Survived"] * 100  # 백분률로 변환
survive_ratio.columns = ["group","sex","survive_rate"]
qs.Lines()
print(survive_ratio)

sns.catplot(x='agegroup', y='Survived', hue='Sex', data=titanic, kind='bar')
qs.MyShow(plt)

survive_rate = titanic.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()
pivot = titanic.pivot_table(
    values="Survived",
    index="Pclass",
    columns="Sex",
    aggfunc="mean",      # 각 그룹의 평균(생존율)
    margins=True,        # '전체 요약값' 행/열 추가 (aggfunc가 다시 원자료 전체에 적용됨)
    margins_name="Total" # 요약 라벨명
)
qs.Lines()
print(pivot)

# 중복 확인 - 다양한 방법
qs.Lines()
print("=== 중복 확인 ===")

# 방법 1: Name과 Ticket 조합으로 중복 확인
duplicates_name_ticket = titanic[titanic.duplicated(subset=["Name","Ticket"])]
print(f"Name+Ticket 중복 행 수: {len(duplicates_name_ticket)}")

# 방법 2: Name만으로 중복 확인
duplicates_name = titanic[titanic.duplicated(subset=["Name"])]
print(f"Name만 중복 행 수: {len(duplicates_name)}")

# 방법 3: Ticket만으로 중복 확인
duplicates_ticket = titanic[titanic.duplicated(subset=["Ticket"])]
print(f"Ticket만 중복 행 수: {len(duplicates_ticket)}")

# 방법 4: 전체 행 중복 확인
duplicates_all = titanic[titanic.duplicated()]
print(f"전체 행 중복 수: {len(duplicates_all)}")

# 방법 5: 중복이 있다면 출력
if len(duplicates_name) > 0:
    qs.Lines()
    print("Name 중복 행들:")
    print(duplicates_name[["Name", "Ticket", "PassengerId"]].head())

if len(duplicates_ticket) > 0:
    qs.Lines()
    print("Ticket 중복 행들:")
    print(duplicates_ticket[["Name", "Ticket", "PassengerId"]].head())



