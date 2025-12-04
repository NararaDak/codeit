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
bmw = pd.read_csv('D:/01.project/코드잇/part-1-main/data/bmw.csv')
print(bmw.head())



# Region-Sales_Volume-Sales_Classification (합계로 표시)

# groupby로 합계 계산 후 시각화
bmw_sum = bmw.groupby(['Region', 'Sales_Classification'])['Sales_Volume'].sum().reset_index()
qs.Lines()
print("Region별 Sales_Classification별 Sales_Volume 합계:")
print(bmw_sum)

# 그래프 크기 설정 (catplot에서만 설정)
g = sns.catplot(x='Region', y='Sales_Volume', hue='Sales_Classification', data=bmw_sum, kind='bar', height=8, aspect=2.5)

# 제목과 축 라벨 설정
g.fig.suptitle('Region별 Sales_Volume 합계 (Sales_Classification별)', fontsize=16, y=0.95)
g.set_axis_labels('Region', 'Sales_Volume', fontsize=14)

# 막대 위에 값 표시
ax = g.axes[0, 0]  # catplot의 axes 가져오기
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            f'{int(height):,}',  # 천 단위 구분자와 함께 표시
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)  # 새로운 범례 생성
plt.tight_layout()  # 레이아웃 자동 조정
plt.show()

bmw_group = bmw.groupby("Region")['Sales_Volume'].sum().reset_index()
qs.Lines()
print(bmw_group.head())
g = sns.catplot(x='Region', y='Sales_Volume',  data=bmw_group, kind='bar',height=8, aspect=2.5)
ax = g.axes[0, 0]  # catplot의 axes 가져오기
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            f'{int(height):,}',  # 천 단위 구분자와 함께 표시
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()

# 모델/년도별 판매량
qs.Lines()
bmw_group = bmw.groupby(["Model","Year"])['Sales_Volume'].sum().reset_index()
print(bmw_group)
g = sns.catplot(x='Year', y='Sales_Volume', hue="Model", data=bmw_group, kind='bar',height=8, aspect=2.5)
plt.show()


# 모델별 판매량
bmw_group = bmw.groupby("Model")['Sales_Volume'].sum().reset_index()
print(bmw_group)
g = sns.catplot(x='Model', y='Sales_Volume',  data=bmw_group, kind='bar',height=8, aspect=2.5)
plt.show()

# fuel_type, Sales_Volume

bmw_group = bmw.groupby("Fuel_Type")['Sales_Volume'].sum().reset_index()
print(bmw_group)
g = sns.catplot(x='Fuel_Type', y='Sales_Volume',  data=bmw_group, kind='bar',height=8, aspect=2.5)
ax = g.axes[0, 0]  # catplot의 axes 가져오기
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            f'{int(height):,}',  # 천 단위 구분자와 함께 표시
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()


# **원 그래프(Pie Chart)로 시각화**
plt.figure(figsize=(10, 10)) # 그래프 크기 설정

# 'Sales_Volume' 값과 'Fuel_Type' 라벨 추출
sizes = bmw_group['Sales_Volume']
labels = bmw_group['Fuel_Type']

# 원 그래프 생성
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%', # 각 조각에 비율을 소수점 첫째 자리까지 표시
    startangle=90,    # 첫 번째 조각이 90도(위쪽)에서 시작하도록 설정
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}, # 각 조각의 테두리 설정
    textprops={'fontsize': 12, 'fontweight': 'bold'} # 텍스트 스타일 설정
)

plt.title('Fuel_Type별 Sales_Volume 비율', fontsize=18)
plt.axis('equal') # 원형을 완벽하게 유지하도록 설정

# 범례를 그래프 밖에 표시 (필요에 따라 주석 처리 가능)
# plt.legend(labels, title="연료 타입", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()


# --- **Engine_Size와 Price의 양의 상관관계 산점도 (추가된 부분)** ---
qs.Lines()
print("Engine_Size와 Price의 양의 상관관계 분석 (산점도)")

plt.figure(figsize=(10, 8))

# 산점도(scatterplot)와 추세선(regression line)을 함께 그리는 regplot 사용
sns.regplot(x='Engine_Size_L', y='Price_USD', data=bmw, scatter_kws={'s': 100, 'alpha': 0.7}, line_kws={'color': 'red'})

plt.title('Engine Size와 Price의 양의 상관관계', fontsize=16)
plt.xlabel('Engine Size (L)', fontsize=14)
plt.ylabel('Price_USD (천)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()