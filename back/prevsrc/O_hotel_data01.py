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

# ================================
# 1. 데이터 로드
# ================================
# 현재 스크립트가 위치한 디렉토리 경로 가져오기
script_dir = os.path.dirname(os.path.abspath(__file__))
# 호텔 데이터 파일의 전체 경로 생성
data_path = os.path.join(script_dir, 'data', 'hotel_data_modified.csv')
# CSV 파일을 DataFrame으로 읽어오기
hotel_data = pd.read_csv(data_path)

# ================================
# 2. 데이터 기본 정보 확인
# ================================
# 데이터의 첫 5행 출력 (데이터 구조 파악)
qs.Lines()
print("=== 데이터 첫 5행 ===")
print(hotel_data.head())
qs.Lines()

# 데이터의 크기 (행, 열 개수) 확인
print("=== 데이터 크기 (행, 열) ===")
print(hotel_data.shape)
qs.Lines()

# 수치형 데이터의 기본 통계 정보 (평균, 표준편차, 최솟값, 최댓값 등)
print("=== 수치형 데이터 기본 통계 ===")
print(hotel_data.describe())
qs.Lines()

# 데이터 타입, 메모리 사용량, null 값 개수 등 상세 정보
print("=== 데이터 상세 정보 ===")
print(hotel_data.info())
qs.Lines()

# 각 컬럼별 결측값(null) 개수 확인
print("=== 컬럼별 결측값 개수 ===")
print(hotel_data.isnull().sum())

# ================================
# 3. 데이터 전처리
# ================================
# 'company' 컬럼 삭제 (분석에 불필요하거나 문제가 있는 컬럼으로 판단)
hotel_data.drop("company", inplace=True, axis=1)
print("=== 'company' 컬럼 삭제 후 데이터 크기 ===")
print(hotel_data.shape)

# ================================
# 4. 결측값 처리 (Missing Value Imputation)
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
# 5. 데이터 타입 변환
# ================================

# 'arrival_date_year' 컬럼을 정수형에서 문자열형으로 변환
# apply(lambda x: str(x)): 각 값을 문자열로 변환
# 날짜 관련 분석이나 그룹핑 시 문자열로 처리하는 것이 더 적절할 수 있음
hotel_data["arrival_date_year"] = hotel_data["arrival_date_year"].apply(lambda x: str(x))


# ================================
# 6. 데이터 시각화 - 호텔 타입별 분포
# ================================

# 그래프 크기 설정 (가로 8, 세로 8 인치)
plt.rcParams["figure.figsize"] = 8, 8

# 호텔 타입별 데이터 개수 계산
# labels: 호텔 타입 이름 리스트 (예: ['City Hotel', 'Resort Hotel'])
labels = hotel_data["hotel"].value_counts().index.tolist()

# sizes: 각 호텔 타입별 데이터 개수 리스트
sizes = hotel_data["hotel"].value_counts().tolist()

# 파이 차트 색상 설정 (주황색, 하늘색)
colors = ["darkorange", "lightskyblue"]

# 파이 차트 생성 및 설정
# sizes: 각 섹션의 크기
# labels: 각 섹션의 라벨
# colors: 각 섹션의 색상
# autopct: 퍼센트 표시 형식 (%1.1f%% = 소수점 첫째자리까지)
# startangle: 시작 각도 (90도에서 시작)
# textprops: 텍스트 속성 (폰트 크기 14)
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 14})

# 그래프 제목 설정
plt.title("호텔 타입별 예약 분포", fontsize=16, fontweight='bold')

# 그래프 표시
plt.show()


# ================================
# 7. 월별 호텔 예약 분포 시각화
# ================================

# 그래프 크기 설정 (가로 20, 세로 5 인치 - 월별 데이터를 넓게 표시)
plt.figure(figsize=(20, 10))

# 분석에 사용할 컬럼 선택 (호텔 타입, 도착 월)
myList = ["hotel", "arrival_date_month"]

# 월별 예약 수를 호텔 타입별로 구분하여 막대 그래프 생성
# data: 분석할 데이터 (호텔 타입과 도착 월 컬럼만 선택)
# x: x축에 표시할 컬럼 (도착 월)
# hue: 색상으로 구분할 컬럼 (호텔 타입)
# order: 월의 순서를 올바르게 정렬
sns.countplot(data=hotel_data[myList], x="arrival_date_month", hue="hotel", order=[
   "January", "February", "March", "April", "May", "June",
   "July", "August", "September", "October", "November", "December"]).set_title(
    "월별 호텔 예약 분포")

# x축 라벨 설정
plt.xlabel("월")
# y축 라벨 설정  
plt.ylabel("예약 수")

# 그래프 표시
plt.show()

# ================================
# 8. 월별 호텔 취소 현황 시각화
# ================================

# 그래프 크기 설정 (가로 20, 세로 5 인치)
plt.figure(figsize=(20, 10))

# 월별, 호텔별, 취소 여부별로 그룹화하여 취소된 예약 수 계산
# groupby로 그룹화 후 size()로 각 그룹의 개수를 계산
hotel_data_cancel = hotel_data.groupby(["arrival_date_month", "hotel", "is_canceled"]).size().reset_index(name="count")

# 취소된 예약만 필터링 (is_canceled == 1)
hotel_data_cancel = hotel_data_cancel[hotel_data_cancel["is_canceled"] == 1]


# 월별 취소 수를 호텔 타입별로 구분하여 막대 그래프 생성
# data: 취소된 예약 데이터
# x: x축에 표시할 컬럼 (도착 월)
# y: y축에 표시할 값 (취소 수)
# hue: 색상으로 구분할 컬럼 (호텔 타입)
# order: 월의 순서를 올바르게 정렬
sns.barplot(data=hotel_data_cancel, x="arrival_date_month", y="count", hue="hotel", order=[
   "January", "February", "March", "April", "May", "June",
   "July", "August", "September", "October", "November", "December"]).set_title(
    "월별 호텔 취소 현황")

# x축 라벨 설정
plt.xlabel("월")
# y축 라벨 설정  
plt.ylabel("취소 수")

# 그래프 표시
plt.show()

# ================================
# 9. 월별 예약수와 취소수 비율 시각화
# ================================
# 그래프 크기 설정 (가로 20, 세로 10 인치)
plt.figure(figsize=(20, 10))

# 월별, 호텔별로 총 예약수와 취소수 계산
hotel_data_summary = hotel_data.groupby(["arrival_date_month", "hotel"]).agg({
    'is_canceled': ['count', 'sum']  # count: 총 예약수, sum: 취소수
}).reset_index()

# 컬럼명 정리 (다중 레벨 컬럼명을 단순화)
hotel_data_summary.columns = ['arrival_date_month', 'hotel', 'total_bookings', 'cancelled_bookings']

# 취소율 계산 (취소수 / 총 예약수 * 100)
hotel_data_summary['cancellation_rate'] = (hotel_data_summary['cancelled_bookings'] / 
                                          hotel_data_summary['total_bookings'] * 100).round(2)

# 취소율을 호텔 타입별로 구분하여 막대 그래프 생성
ax = sns.barplot(data=hotel_data_summary, x="arrival_date_month", y="cancellation_rate", 
           hue="hotel", order=[
               "January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]).set_title(
               "월별 호텔 취소율 (%)")

# 막대 그래프 위에 숫자 값 표시
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.1f', fontsize=8 )

# x축 라벨 설정
plt.xlabel("월")
# y축 라벨 설정  
plt.ylabel("취소율 (%)")

# 그래프 표시
plt.show()

# ================================
# 10. 월별/호텔별 연령대별 고객 수 집계 (취소 제외)
# ================================

# 취소되지 않은 예약만 필터링 (is_canceled == 0)
hotel_data_not_cancelled = hotel_data[hotel_data["is_canceled"] == 0]

# 월별, 호텔별로 그룹화하여 adults, children, babies 합계 계산
hotel_age_summary = hotel_data_not_cancelled.groupby(["arrival_date_month", "hotel"]).agg({
    'adults': 'sum',      # 성인 수 합계
    'children': 'sum',    # 아동 수 합계  
    'babies': 'sum'       # 유아 수 합계
}).reset_index()

# 컬럼명을 한글로 변경하여 가독성 향상
hotel_age_summary.columns = ['월', '호텔', '성인수', '아동수', '유아수']

# 월 순서를 올바르게 정렬
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# 월 순서대로 데이터 정렬
hotel_age_summary['월'] = pd.Categorical(hotel_age_summary['월'], categories=month_order, ordered=True)
hotel_age_summary = hotel_age_summary.sort_values(['월', '호텔'])

# 결과 출력
print("=== 월별/호텔별 연령대별 고객 수 집계 (취소 제외) ===")
qs.Lines()
print(hotel_age_summary)

# 시각화를 위한 데이터 준비 (월별 순서대로)
hotel_age_summary_viz = hotel_age_summary.copy()
hotel_age_summary_viz['월'] = hotel_age_summary_viz['월'].astype(str)

# 그래프 크기 설정
plt.figure(figsize=(20, 10))

# 서브플롯 생성 (3개 그래프를 세로로 배치)
fig, axes = plt.subplots(3, 1, figsize=(20, 15))

# 1. 성인 수 막대 그래프
sns.barplot(data=hotel_age_summary_viz, x="월", y="성인수", hue="호텔", ax=axes[0], order=month_order)
axes[0].set_title("월별 호텔 성인 고객 수 (취소 제외)", fontsize=14, fontweight='bold')
axes[0].set_xlabel("월")
axes[0].set_ylabel("성인 수")

# 막대 위에 숫자 표시
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%d', fontsize=8)

# 2. 아동 수 막대 그래프
sns.barplot(data=hotel_age_summary_viz, x="월", y="아동수", hue="호텔", ax=axes[1], order=month_order)
axes[1].set_title("월별 호텔 아동 고객 수 (취소 제외)", fontsize=14, fontweight='bold')
axes[1].set_xlabel("월")
axes[1].set_ylabel("아동 수")

# 막대 위에 숫자 표시
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%d', fontsize=8)

# 3. 유아 수 막대 그래프
sns.barplot(data=hotel_age_summary_viz, x="월", y="유아수", hue="호텔", ax=axes[2], order=month_order)
axes[2].set_title("월별 호텔 유아 고객 수 (취소 제외)", fontsize=14, fontweight='bold')
axes[2].set_xlabel("월")
axes[2].set_ylabel("유아 수")

# 막대 위에 숫자 표시
for container in axes[2].containers:
    axes[2].bar_label(container, fmt='%d', fontsize=8)

# 그래프 간 간격 조정
plt.tight_layout()

# 그래프 표시
plt.show()

# ================================
# 11. 연령대별 고객 수 통합 시각화 (하나의 그래프)
# ================================

# 라인 차트 (연령대별 추이)
plt.figure(figsize=(20, 8))

# 월별 순서대로 정렬
hotel_age_sorted = hotel_age_summary_viz.copy()
hotel_age_sorted['월'] = pd.Categorical(hotel_age_sorted['월'], categories=month_order, ordered=True)
hotel_age_sorted = hotel_age_sorted.sort_values(['월', '호텔'])

# City Hotel과 Resort Hotel 데이터 분리
city_data = hotel_age_sorted[hotel_age_sorted['호텔'] == 'City Hotel']
resort_data = hotel_age_sorted[hotel_age_sorted['호텔'] == 'Resort Hotel']

# 라인 차트 생성
plt.plot(city_data['월'], city_data['성인수'], marker='o', linewidth=2, label='City Hotel - 성인', color='#FF6B6B')
plt.plot(city_data['월'], city_data['아동수'], marker='s', linewidth=2, label='City Hotel - 아동', color='#4ECDC4')
plt.plot(city_data['월'], city_data['유아수'], marker='^', linewidth=2, label='City Hotel - 유아', color='#45B7D1')

plt.plot(resort_data['월'], resort_data['성인수'], marker='o', linewidth=2, linestyle='--', label='Resort Hotel - 성인', color='#FF6B6B', alpha=0.7)
plt.plot(resort_data['월'], resort_data['아동수'], marker='s', linewidth=2, linestyle='--', label='Resort Hotel - 아동', color='#5ECDC4', alpha=0.7)
plt.plot(resort_data['월'], resort_data['유아수'], marker='^', linewidth=2, linestyle='--', label='Resort Hotel - 유아', color='#45B7D1', alpha=0.7)

plt.title('월별 연령대별 고객 수 추이 (취소 제외)', fontsize=16, fontweight='bold')
plt.xlabel('월')
plt.ylabel('고객 수')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ================================
# 12. 통합 히트맵 - 나라별 셀 내부에서 호텔 타입 구분
# ================================

# 데이터 준비: 나라별로 City Hotel과 Resort Hotel 데이터를 합치기
# 1. 각 호텔별 데이터에 호텔 타입 정보 추가
city_data = hotel_data[hotel_data['hotel'] == 'City Hotel'].copy()
resort_data = hotel_data[hotel_data['hotel'] == 'Resort Hotel'].copy()

# 2. 나라별, 월별 예약 수 계산 (취소 제외)
city_monthly = city_data[city_data['is_canceled'] == 0].groupby(['country', 'arrival_date_month']).size().reset_index(name='bookings')
resort_monthly = resort_data[resort_data['is_canceled'] == 0].groupby(['country', 'arrival_date_month']).size().reset_index(name='bookings')

# 3. 호텔 타입 정보 추가
city_monthly['hotel_type'] = 'City'
resort_monthly['hotel_type'] = 'Resort'

# 4. 데이터 합치기
combined_data = pd.concat([city_monthly, resort_monthly], ignore_index=True)

# 5. 피벗 테이블 생성 (나라 x 월 x 호텔타입)
pivot_combined = combined_data.pivot_table(
    index='country', 
    columns='arrival_date_month', 
    values='bookings',
    fill_value=0
)

# 6. 월 순서대로 컬럼 정렬
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
pivot_combined = pivot_combined.reindex(columns=month_order, fill_value=0)

# 7. 나라별로 City와 Resort 데이터 분리
countries = pivot_combined.index.tolist()
months = pivot_combined.columns.tolist()

# 8. 새로운 데이터프레임 생성 (각 셀을 2개로 분할)
# 각 나라-월 조합에 대해 City(위), Resort(아래) 데이터를 담을 구조
expanded_data = []
for country in countries:
    for month in months:
        city_val = city_monthly[(city_monthly['country'] == country) & 
                               (city_monthly['arrival_date_month'] == month)]['bookings'].sum()
        resort_val = resort_monthly[(resort_monthly['country'] == country) & 
                                   (resort_monthly['arrival_date_month'] == month)]['bookings'].sum()
        
        expanded_data.append({
            'country': country,
            'month': month,
            'city_bookings': city_val,
            'resort_bookings': resort_val
        })

expanded_df = pd.DataFrame(expanded_data)

# 9. 시각화를 위한 데이터 준비
fig, ax = plt.subplots(figsize=(20, 12))

# 나라별, 월별로 City와 Resort 데이터를 시각화
# 각 셀을 2개로 나누어 위(City), 아래(Resort)로 표시

# 데이터를 피벗하여 시각화용으로 변환
city_pivot = expanded_df.pivot(index='country', columns='month', values='city_bookings')
resort_pivot = expanded_df.pivot(index='country', columns='month', values='resort_bookings')

# 월 순서대로 정렬
city_pivot = city_pivot.reindex(columns=month_order, fill_value=0)
resort_pivot = resort_pivot.reindex(columns=month_order, fill_value=0)

# 히트맵 생성
# City Hotel 히트맵 (위쪽 반)
sns.heatmap(city_pivot, 
            annot=True, 
            fmt='.0f', 
            cmap='Oranges', 
            cbar_kws={'label': 'City Hotel 예약 수'},
            ax=ax,
            annot_kws={"size": 8})

# Resort Hotel 히트맵 (아래쪽 반)을 위한 추가 작업
# 현재는 단순히 City Hotel만 표시되므로, 다른 방법으로 접근

plt.title('나라별/월별 호텔 예약 수 - City Hotel', fontsize=16, fontweight='bold')
plt.xlabel('도착 월 (Arrival Month)')
plt.ylabel('나라 (Country)')

plt.tight_layout()
plt.show()

# ================================
# 13. 대안: 나라별 통합 히트맵 (City + Resort 합계)
# ================================

# 나라별, 월별 총 예약 수 계산 (취소 제외)
total_monthly = hotel_data[hotel_data['is_canceled'] == 0].groupby(['country', 'arrival_date_month']).size().reset_index(name='total_bookings')

# 피벗 테이블 생성
total_pivot = total_monthly.pivot(index='country', columns='arrival_date_month', values='total_bookings')

# 월 순서대로 정렬
total_pivot = total_pivot.reindex(columns=month_order, fill_value=0)

# 히트맵 생성
plt.figure(figsize=(20, 12))
sns.heatmap(total_pivot, 
            annot=True, 
            fmt='.0f', 
            cmap='viridis', 
            cbar_kws={'label': '총 예약 수 (City + Resort)'},
            annot_kws={"size": 8})

plt.title('나라별/월별 총 호텔 예약 수 (City + Resort 통합)', fontsize=16, fontweight='bold')
plt.xlabel('도착 월 (Arrival Month)')
plt.ylabel('나라 (Country)')

plt.tight_layout()
plt.show()

# ================================
# 14. 대안: 나라별 City vs Resort 비교 막대 그래프
# ================================

# 상위 20개 나라 선택 (예약 수 기준)
top_countries = hotel_data[hotel_data['is_canceled'] == 0]['country'].value_counts().head(20).index

# 상위 나라들의 City vs Resort 예약 수 비교
comparison_data = []
for country in top_countries:
    city_count = city_data[(city_data['country'] == country) & (city_data['is_canceled'] == 0)].shape[0]
    resort_count = resort_data[(resort_data['country'] == country) & (resort_data['is_canceled'] == 0)].shape[0]
    
    comparison_data.append({
        'country': country,
        'city_bookings': city_count,
        'resort_bookings': resort_count,
        'total': city_count + resort_count
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('total', ascending=True)

# 막대 그래프 생성
plt.figure(figsize=(16, 10))

# City Hotel 막대 (왼쪽)
plt.barh(range(len(comparison_df)), comparison_df['city_bookings'], 
         label='City Hotel', color='orange', alpha=0.8)

# Resort Hotel 막대 (City 막대 위에 누적)
plt.barh(range(len(comparison_df)), comparison_df['resort_bookings'], 
         left=comparison_df['city_bookings'], label='Resort Hotel', color='blue', alpha=0.8)

# 나라 이름 표시
plt.yticks(range(len(comparison_df)), comparison_df['country'])
plt.xlabel('예약 수')
plt.title('상위 20개 나라별 호텔 타입별 예약 수 비교 (취소 제외)', fontsize=16, fontweight='bold')
plt.legend()

# 각 막대 위에 총 예약 수 표시
for i, (idx, row) in enumerate(comparison_df.iterrows()):
    plt.text(row['total'] + 50, i, f"{row['total']}", va='center', fontsize=8)

plt.tight_layout()
plt.show()
plt.figure(figsize=(10,10))

# ================================
# 15. 나라별/월별 호텔 예약 수 히트맵 시각화
#     (순위별 25% 단위 색상 구분)
# ================================

# 취소되지 않은 예약만 사용
hotel_data_not_cancelled = hotel_data[hotel_data["is_canceled"] == 0]

# 월 순서 정의 (시각화 시 순서 유지를 위해)
month_order = ["January", "February", "March", "April", "May", "June","July", "August", "September", "October", "November", "December"]


# 1. 호텔 타입별 피벗 테이블 생성 (행: country, 열: arrival_date_month, 값: 예약 수)
# City Hotel 데이터
city_hotel_pivot = hotel_data_not_cancelled[ hotel_data_not_cancelled['hotel'] == 'City Hotel'
].pivot_table(
    index='country', 
    columns='arrival_date_month', 
    values='is_canceled', # 예약 건수 (개수를 세는 용도로 사용)
    aggfunc='count'
)

# Resort Hotel 데이터
resort_hotel_pivot = hotel_data_not_cancelled[
    hotel_data_not_cancelled['hotel'] == 'Resort Hotel'
].pivot_table(
    index='country', 
    columns='arrival_date_month', 
    values='is_canceled', 
    aggfunc='count'
)

# 월 순서대로 컬럼 정렬 (없는 월은 NaN으로 채워짐)
city_hotel_pivot = city_hotel_pivot.reindex(columns=month_order)
resort_hotel_pivot = resort_hotel_pivot.reindex(columns=month_order)

# 결측값(NaN)은 예약이 없다는 뜻이므로 0으로 대체
city_hotel_pivot = city_hotel_pivot.fillna(0)
resort_hotel_pivot = resort_hotel_pivot.fillna(0)


# 2. 순위(예약 수) 기반 색상 분할 함수 정의
def get_color_map(data):
    """
    데이터의 예약 수를 4분위수(25% 단위)를 기준으로 나누어 색상 맵을 생성
    
    Parameters:
    data (pd.DataFrame): 예약 수가 포함된 피벗 테이블
    
    Returns:
    list: 색상 코드 리스트
    """
    # 전체 예약 수 데이터 평탄화 (1차원 배열로 만듦)
    all_values = data.values.flatten()
    
    # 0인 값은 제외하고 순위 분할
    non_zero_values = all_values[all_values > 0]
    
    if len(non_zero_values) == 0:
        # 데이터가 모두 0인 경우 (색상 구분 불필요)
        return ['#F0F0F0'] * 4 

    # 4분위수(Quartiles) 계산: 0%, 25%, 50%, 75%, 100%
    # q1: 25%, q2: 50%, q3: 75%
    q1, q2, q3 = np.percentile(non_zero_values, [25, 50, 75])

    # 색상 정의 (순위가 높을수록 진한 색)
    # [0, 25%] - 가장 낮은 25%
    # (25%, 50%] - 다음 25%
    # (50%, 75%] - 다음 25%
    # (75%, 100%] - 가장 높은 25%
    
    # Matplotlib의 컬러맵(Oranges)을 사용하여 색상 농도를 조절합니다.
    # 25% 구간별로 색상을 다르게 지정합니다.
    def value_to_color(val):
        if val == 0:
            return '#F0F0F0' # 예약이 없는 경우 (매우 밝은 회색)
        elif val <= q1:
            return '#FEE5D9' # 0-25% (가장 연한 주황)
        elif val <= q2:
            return '#FCAE91' # 25-50% (연한 주황)
        elif val <= q3:
            return '#FB6A4A' # 50-75% (중간 주황)
        else: # val > q3
            return '#CB181D' # 75-100% (가장 진한 빨강)

    # 데이터프레임의 각 요소에 색상 매핑 함수 적용
    return data.applymap(value_to_color)


# 3. 색상 매핑 적용
city_colors = get_color_map(city_hotel_pivot)
resort_colors = get_color_map(resort_hotel_pivot)


# 4. 시각화 (두 개의 서브플롯)
plt.rcParams["figure.figsize"] = 18, 15
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 15), sharex=True)

# City Hotel 히트맵 (상단)
# 'annot=True'는 셀 안에 예약 수를 표시
# 'fmt='는 정수 형태로 표시
# 'cbar=False'는 컬러 바를 표시하지 않음 (색상이 분위수별로 분할되어 cbar가 의미 없음)
sns.heatmap(
    city_hotel_pivot, 
    annot=True, 
    fmt='.0f', 
    linewidths=.5, 
    cmap='Oranges', 
    cbar=False,
    ax=axes[0],
    linecolor='gray',
    annot_kws={"size": 8}
)
axes[0].set_title('City Hotel - 나라별/월별 예약 수 (순위 25% 단위 색상 구분)', fontsize=16, fontweight='bold')
axes[0].set_ylabel('나라 (Country)')
axes[0].set_xlabel('') # 상단 그래프는 x축 라벨 생략

# Resort Hotel 히트맵 (하단)
sns.heatmap(
    resort_hotel_pivot, 
    annot=True, 
    fmt='.0f', 
    linewidths=.5, 
    cmap='Blues', 
    cbar=False,
    ax=axes[1],
    linecolor='gray',
    annot_kws={"size": 8}
)
axes[1].set_title('Resort Hotel - 나라별/월별 예약 수 (순위 25% 단위 색상 구분)', fontsize=16, fontweight='bold')
axes[1].set_ylabel('나라 (Country)')
axes[1].set_xlabel('도착 월 (Arrival Month)')


# 컬러 설명 추가 (범례 역할을 대신)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FEE5D9', edgecolor='black', label='0-25% (Low Bookings)'),
    Patch(facecolor='#FCAE91', edgecolor='black', label='25-50%'),
    Patch(facecolor='#FB6A4A', edgecolor='black', label='50-75%'),
    Patch(facecolor='#CB181D', edgecolor='black', label='75-100% (High Bookings)'),
    Patch(facecolor='#F0F0F0', edgecolor='black', label='0 Bookings')
]
fig.legend(handles=legend_elements, title='예약 수 순위', loc='lower center', ncol=5, 
           bbox_to_anchor=(0.5, -0.05), fontsize=10, title_fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 1]) # 범례 공간 확보
plt.show()

