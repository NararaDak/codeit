# 필요한 라이브러리 import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import lib.QS as qs


# ================================
# 그래프 설정
# ================================
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')  # 그래프 스타일 설정


# 화면 지우기 (출력 결과를 깔끔하게 하기 위해)
qs.MyClear()


# 화면 지우기 (출력 결과를 깔끔하게 하기 위해)
qs.MyClear()

# ================================
# 데이터 로드
# ================================
# 현재 스크립트가 위치한 디렉토리 경로 가져오기
script_dir = os.path.dirname(os.path.abspath(__file__))
# 호텔 데이터 파일의 전체 경로 생성
data_path = os.path.join(script_dir, 'data', 'hotel_data_modified.csv')
# CSV 파일을 DataFrame으로 읽어오기
hotel_data = pd.read_csv(data_path)




# ================================
# 결측값 처리 (Missing Value Imputation)
# ================================
def impute_median(series):
    """
    결측값을 해당 시리즈의 중앙값으로 대체하는 함수
    
    Parameters:
    series (pandas.Series): 결측값이 포함된 데이터 시리즈
    
    Returns:
    pandas.Series: 중앙값으로 결측값이 채워진 시리즈
    """
    return series.fillna(series.median())

# 'children' 컬럼의 결측값을 중앙값으로 대체
hotel_data.children = hotel_data["children"].transform(impute_median)

# 'agent' 컬럼의 결측값을 중앙값으로 대체  
hotel_data.agent = hotel_data["agent"].transform(impute_median)

# 결측값 처리 후 각 컬럼별 결측값 개수 재확인
print("=== 결측값 처리 후 각 컬럼별 결측값 개수 ===")
print(hotel_data.isnull().sum())


# 'country' 컬럼의 최빈값(가장 자주 나타나는 값) 확인
print("=== 'country' 컬럼의 최빈값 ===")
print(hotel_data["country"].mode())

# 'country' 컬럼의 결측값을 최빈값으로 대체
# mode().values[0]: 최빈값을 numpy 배열 형태로 가져온 후 첫 번째 값 선택
# str(): 문자열 형태로 변환하여 결측값 대체
hotel_data["country"].fillna(str(hotel_data["country"].mode().values[0]), inplace=True)

# 최종 결측값 처리 후 각 컬럼별 결측값 개수 최종 확인
print("=== 최종 결측값 처리 후 각 컬럼별 결측값 개수 ===")
print(hotel_data.isnull().sum())

# ================================
# 데이터 타입 변환
# ================================
# 'arrival_date_year' 컬럼을 정수형에서 문자열형으로 변환
# apply(lambda x: str(x)): 각 값을 문자열로 변환
# 날짜 관련 분석이나 그룹핑 시 문자열로 처리하는 것이 더 적절할 수 있음
hotel_data["arrival_date_year"] = hotel_data["arrival_date_year"].apply(lambda x: str(x))



# ================================
# Lead Time 데이터 품질 분석
# ================================

# Lead Time 기본 통계 확인
print("=== Lead Time 기본 통계 ===")
qs.Lines()
print(hotel_data['lead_time'].describe())
qs.Lines()


# Lead Time이 0인 경우 확인
zero_lead_time = hotel_data[hotel_data['lead_time'] == 0]
print(f"=== Lead Time이 0인 레코드 수: {len(zero_lead_time)} ===")
if len(zero_lead_time) > 0:
    print("0 Lead Time 샘플 데이터:")
    print(zero_lead_time[['hotel', 'arrival_date_year', 'arrival_date_month', 'lead_time', 'is_canceled']].head(10))
    qs.Lines()

# Lead Time이 매우 큰 값인 경우 확인 (이상치)
high_lead_time = hotel_data[hotel_data['lead_time'] > 365]  # 1년 이상
print(f"=== Lead Time이 365일 이상인 레코드 수: {len(high_lead_time)} ===")
if len(high_lead_time) > 0:
    print("높은 Lead Time 샘플 데이터:")
    print(high_lead_time[['hotel', 'arrival_date_year', 'arrival_date_month', 'lead_time', 'is_canceled']].head(10))
    qs.Lines()

# Lead Time 분포 히스토그램
plt.figure(figsize=(10, 5))

# 전체 Lead Time 분포
plt.subplot(1, 2, 1)
plt.hist(hotel_data['lead_time'], bins=50, alpha=0.7, edgecolor='black')
plt.title('전체 Lead Time 분포')
plt.xlabel('Lead Time (일)')
plt.ylabel('빈도')
plt.axvline(x=0, color='red', linestyle='--', label='Lead Time = 0')
plt.legend()


# 0-30일 범위의 Lead Time
plt.subplot(1, 2, 2)
normal_data = hotel_data[(hotel_data['lead_time'] >= 0) & (hotel_data['lead_time'] <= 30)]['lead_time']
plt.hist(normal_data, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title('정상 범위 Lead Time (0-30일)')
plt.xlabel('Lead Time (일)')
plt.ylabel('빈도')

plt.tight_layout()
plt.show()

# ================================
# 바이올린 플롯 - 도착 연도별 리드 타임 분포 분석
# ================================

# 그래프 크기 설정 (10x10 인치의 정사각형)
plt.figure(figsize=(10, 10))

# 바이올린 플롯 생성
# x: 도착 연도 (arrival_date_year) - x축에 표시할 카테고리
# y: 리드 타임 (lead_time) - y축에 표시할 수치 데이터
# hue: 취소 여부 (is_canceled) - 색상으로 구분할 카테고리 (0: 취소 안함, 1: 취소함)
# data: 분석할 데이터프레임
# palette: 색상 팔레트 ("Set3" - 다양한 색상 조합)
# bw: 대역폭 조정 (2로 설정하여 분포의 부드러움 조절)
# linewidth: 바이올린 테두리 두께 (2로 설정하여 명확한 구분)
# inner: 바이올린 내부에 박스 플롯 표시 ("box" - 중앙값, 사분위수 등 표시)
# split: 취소 여부에 따라 바이올린을 분할하여 표시 (True - 같은 연도에서 취소/비취소 비교 가능)
sns.violinplot(x="arrival_date_year", y="lead_time", hue="is_canceled", data=hotel_data, 
               palette="Set3", bw=2, linewidth=2, inner="box", split=True)

# 왼쪽 축 제거 (깔끔한 시각화를 위해)
sns.despine(left=True)

# 그래프 제목 설정
plt.title("Arrival Year vs Lead Time vs Canceled Situation", weight="bold")

# x축 라벨 설정 (도착 연도)
plt.xlabel("Year", fontsize=12)

# y축 라벨 설정 (리드 타임)
plt.ylabel("Lead Time", fontsize=12)

# 레이아웃 자동 조정 (텍스트가 겹치지 않도록)
plt.tight_layout()

# 그래프 표시
plt.show()



hotel_data['arrival_date_month'].replace({'January' : '1',
        'February' : '2',
        'March' : '3',
        'April' : '4',
        'May' : '5',
        'June' : '6',
        'July' : '7',
        'August' : '8',
        'September' : '9', 
        'October' : '10',
        'November' : '11',
        'December' : '12'}, inplace=True)








# 6. Lead Time 구간별 취소율 분석
print("\n=== 6. Lead Time 구간별 취소율 분석 시작 ===")

# Lead Time을 범주형 구간으로 나누기
# 0일, 1-30일, 31-90일, 91-180일, 181-365일, 366일 이상
bins = [-1, 0, 30, 90, 180, 365, hotel_data['lead_time'].max()]
labels = ['0일(당일)', '1-30일', '31-90일', '91-180일', '181-365일', '366일 이상']

# Lead Time의 최댓값이 365보다 크므로 마지막 bin을 동적으로 설정
if hotel_data['lead_time'].max() > 365:
    bins[-1] = hotel_data['lead_time'].max() + 1 # max() 값도 포함하기 위해 +1

# right=True: (a, b] 형태로, 0일 구간을 포함하기 위해 -1부터 시작
hotel_data['lead_time_group'] = pd.cut(hotel_data['lead_time'], bins=bins, labels=labels, right=True)

# 구간별 취소율 계산
# observed=False: 사용하지 않은 범주도 표시 (선택적으로 사용)
cancel_rate_by_lead_time = hotel_data.groupby('lead_time_group', observed=False)['is_canceled'].mean() * 100

plt.figure(figsize=(14, 7))
ax6 = sns.barplot(x=cancel_rate_by_lead_time.index, y=cancel_rate_by_lead_time.values, palette='viridis')
plt.title('6. 예약 선행 기간(Lead Time) 구간별 취소율', fontsize=18)
plt.xlabel('Lead Time 구간 (일)', fontsize=14)
plt.ylabel('취소율 (%)', fontsize=14)
plt.xticks(rotation=45) # x축 레이블 회전

# 각 막대 위에 값 표시
for p in ax6.patches:
    height = p.get_height()
    ax6.text(p.get_x() + p.get_width() / 2.,
            height + 0.5,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()

print("\n=== 6. Lead Time 구간별 취소율 분석 완료 ===")

