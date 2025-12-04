# 필요한 라이브러리 import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

# ================================
# 그래프 설정
# ================================
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')  # 그래프 스타일 설정
plt.rcParams["figure.figsize"] = 10, 6 # 기본 그래프 크기 설정

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
print("\n" + "="*50)
print("=== [2.1] 데이터 첫 5행 ===")
print(hotel_data.head())
print("\n" + "="*50)
print("=== [2.2] 데이터 크기 (행, 열) ===")
print(hotel_data.shape)
print("\n" + "="*50)
print("=== [2.3] 수치형 데이터 기본 통계 ===")
print(hotel_data.describe())
print("\n" + "="*50)
print("=== [2.4] 데이터 상세 정보 (타입, Non-Null Count) ===")
print(hotel_data.info())
print("\n" + "="*50)
print("=== [2.5] 컬럼별 결측값(null) 개수 확인 ===")
print(hotel_data.isnull().sum())
print("="*50)


# ================================
# 3. 데이터 전처리 (결측값 및 불필요 컬럼 처리)
# ================================

# 'company' 컬럼 삭제 (결측치가 너무 많고 분석에 불필요)
hotel_data.drop("company", inplace=True, axis=1)
print(f"✅ 'company' 컬럼 삭제 후 데이터 크기: {hotel_data.shape}")

# 결측값 처리 함수 정의 (중앙값 대체)
def impute_median(series):
    return series.fillna(series.median())

# 'children' 및 'agent' 컬럼 결측값을 중앙값으로 대체
hotel_data['children'] = hotel_data['children'].transform(impute_median)
hotel_data['agent'] = hotel_data['agent'].transform(impute_median)

# 'country' 컬럼의 결측값을 최빈값으로 대체
mode_country = hotel_data["country"].mode().values[0]
hotel_data["country"].fillna(mode_country, inplace=True)

# 데이터 타입 변환: 'arrival_date_year' 정수형 -> 문자열형
hotel_data["arrival_date_year"] = hotel_data["arrival_date_year"].astype(str)

print("\n" + "="*50)
print("=== [3.1] 최종 결측값 처리 후 각 컬럼별 결측값 개수 ===")
print(hotel_data.isnull().sum().loc[lambda x: x>0]) # 결측치가 남아있는 컬럼만 출력
if hotel_data.isnull().sum().sum() == 0:
    print("🎉 모든 결측치 처리가 완료되었습니다.")
print("="*50)

# ================================
# 4. EDA 시각화
# ================================
# 월 순서 정의
month_order = [
   "January", "February", "March", "April", "May", "June",
   "July", "August", "September", "October", "November", "December"
]

## 4.1 호텔 타입별 예약 분포 (파이 차트)
plt.figure(figsize=(8, 8))
hotel_counts = hotel_data["hotel"].value_counts()
labels = hotel_counts.index.tolist()
sizes = hotel_counts.tolist()
colors = ["darkorange", "lightskyblue"]

plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 14, "fontweight": 'bold'})
plt.title("호텔 타입별 전체 예약 분포", fontsize=16, fontweight='bold')
plt.show()
print("분석: City Hotel이 Resort Hotel보다 약 1.7배 많은 예약을 차지합니다.")


## 4.2 월별 호텔 예약 분포 (막대 그래프)
plt.figure(figsize=(15, 6))
sns.countplot(data=hotel_data, x="arrival_date_month", hue="hotel", order=month_order)
plt.title("월별 호텔 예약 분포", fontsize=16, fontweight='bold')
plt.xlabel("도착 월")
plt.ylabel("총 예약 수")
plt.xticks(rotation=45)
plt.legend(title='호텔 타입')
plt.tight_layout()
plt.show()
print("분석: City Hotel은 **7월과 8월**에 예약이 가장 많으며, Resort Hotel은 **8월**에 가장 많습니다. 겨울철(11월~1월)에 예약이 감소하는 경향을 보입니다.")


## 4.3 월별 호텔 취소율 (%)
plt.figure(figsize=(15, 6))

# 월별, 호텔별로 총 예약수와 취소수 계산
hotel_data_summary = hotel_data.groupby(["arrival_date_month", "hotel"]).agg({
    'is_canceled': ['count', 'sum']
}).reset_index()

hotel_data_summary.columns = ['arrival_date_month', 'hotel', 'total_bookings', 'cancelled_bookings']
hotel_data_summary['cancellation_rate'] = (hotel_data_summary['cancelled_bookings'] / 
                                          hotel_data_summary['total_bookings'] * 100).round(2)

ax = sns.barplot(data=hotel_data_summary, x="arrival_date_month", y="cancellation_rate", 
           hue="hotel", order=month_order)
ax.set_title("월별 호텔 취소율 (%)", fontsize=16, fontweight='bold')
plt.xlabel("도착 월")
plt.ylabel("취소율 (%)")
plt.xticks(rotation=45)

# 막대 그래프 위에 숫자 값 표시
for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.1f%%', fontsize=8 )

plt.tight_layout()
plt.show()
print("분석: City Hotel의 취소율이 전반적으로 Resort Hotel보다 높습니다. 특히 **4월, 5월, 6월**에 취소율이 높게 나타납니다.")


## 4.4 국가별 대륙 분류 및 예약 분포 분석

# 국가 코드를 대륙으로 매핑하는 함수
def get_continent(country_code):
    """
    국가 코드를 대륙으로 변환하는 함수
    
    Parameters:
    country_code (str): ISO 국가 코드 (예: 'PRT', 'GBR', 'FRA')
    
    Returns:
    str: 대륙명
    """
    # 유럽 국가들
    europe = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'NLD', 'CHE', 'AUT', 'SWE', 'NOR', 'DNK', 
              'FIN', 'POL', 'CZE', 'HUN', 'GRC', 'TUR', 'RUS', 'UKR', 'ROU', 'BGR', 'HRV', 'SVN', 'SVK', 
              'LTU', 'LVA', 'EST', 'LUX', 'ISL', 'MLT', 'CYP']
    
    # 아시아 국가들
    asia = ['CHN', 'JPN', 'KOR', 'IND', 'THA', 'SGP', 'MYS', 'IDN', 'PHL', 'VNM', 'TWN', 'HKG', 'MAC', 'MMR', 
            'KHM', 'LAO', 'BGD', 'PAK', 'LKA', 'NPL', 'BTN', 'MNG', 'KAZ', 'UZB', 'KGZ', 'TJK', 'TKM']
    
    # 아메리카 국가들
    americas = ['USA', 'CAN', 'MEX', 'BRA', 'ARG', 'CHL', 'COL', 'PER', 'VEN', 'ECU', 'BOL', 'PRY', 'URY', 
                'CRI', 'PAN', 'GTM', 'HND', 'NIC', 'SLV', 'BLZ', 'JAM', 'TTO', 'BHS', 'DOM', 'CUB', 'HTI']
    
    # 아프리카 국가들
    africa = ['ZAF', 'EGY', 'MAR', 'TUN', 'DZA', 'LBY', 'SDN', 'ETH', 'KEN', 'GHA', 'NGA', 'SEN', 'CMR', 
              'CIV', 'MAD', 'UGA', 'TZA', 'ZWE', 'ZMB', 'MWI', 'BWA', 'NAM', 'MOZ', 'AGO', 'GAB', 'GNQ']
    
    # 오세아니아 국가들
    oceania = ['AUS', 'NZL', 'FJI', 'PNG', 'TON', 'WSM', 'VUT', 'SLB', 'KIR', 'NCL', 'PYF', 'COK', 'TUV']
    
    # 중동 국가들
    middle_east = ['SAU', 'ARE', 'QAT', 'KWT', 'BHR', 'OMN', 'YEM', 'JOR', 'LBN', 'SYR', 'IRQ', 'IRN', 'AFG']
    
    if country_code in europe:
        return '유럽'
    elif country_code in asia:
        return '아시아'
    elif country_code in americas:
        return '아메리카'
    elif country_code in africa:
        return '아프리카'
    elif country_code in oceania:
        return '오세아니아'
    elif country_code in middle_east:
        return '중동'
    else:
        return '기타'

# 호텔 데이터에 대륙 정보 추가
hotel_data['continent'] = hotel_data['country'].apply(get_continent)

# 대륙별 예약 분포 확인
print("=== 대륙별 예약 분포 (취소 제외) ===")
continent_data = hotel_data[hotel_data["is_canceled"] == 0]
continent_counts = continent_data['continent'].value_counts()
print(continent_counts)
print()

# 대륙별 호텔 타입별 예약 분포
continent_hotel_counts = continent_data.groupby(['continent', 'hotel']).size().unstack(fill_value=0)
print("=== 대륙별 호텔 타입별 예약 분포 ===")
print(continent_hotel_counts)
print()

## 4.4.1 대륙별 예약 분포 시각화
plt.figure(figsize=(15, 10))

# 서브플롯 생성 (2x2)
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# 1. 대륙별 총 예약 수 (파이 차트)
ax1 = axes[0, 0]

# 퍼센트 계산
total_count = continent_counts.sum()
percentages = (continent_counts.values / total_count * 100).round(1)

# 파이 차트 생성 (autopct 제거하여 그래프에 숫자 표시 안함)
wedges, texts = ax1.pie(continent_counts.values, 
                                   labels=None,  # 라벨을 None으로 설정
                                   autopct=None,  # 퍼센트 텍스트 제거
                                   startangle=90,
                                   colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#FFD700', '#C0C0C0'])

# 말풍선 스타일의 주석 추가 (각 섹션에 숫자 표시)
bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
kw = dict(arrowprops=dict(arrowstyle="-", color="black", lw=0.5), 
          bbox=bbox_props, zorder=0, va="center", fontsize=10, fontweight='bold')

for i, (wedge, count) in enumerate(zip(wedges, continent_counts.values)):
    # 각 섹션의 중심각 계산
    ang = (wedge.theta2 + wedge.theta1) / 2
    # 중심각을 라디안으로 변환
    x = np.cos(np.deg2rad(ang))
    y = np.sin(np.deg2rad(ang))
    
    # 말풍선 위치 계산 (원 밖으로)
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    
    # 말풍선으로 숫자 표시
    ax1.annotate(f'{count:,}건', xy=(x, y), xytext=(1.2*x, 1.2*y),
                horizontalalignment=horizontalalignment, **kw)

ax1.set_title('대륙별 예약 분포 (취소 제외)', fontsize=14, fontweight='bold')

# 범례 추가 (대륙 이름과 퍼센트 표시)
legend_labels = [f'{continent} ({percent}%)' for continent, percent in zip(continent_counts.index, percentages)]
ax1.legend(wedges, legend_labels, title="대륙별 예약 비율", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# 2. 대륙별 호텔 타입별 예약 수 (막대 그래프)
ax2 = axes[0, 1]
continent_hotel_counts.plot(kind='bar', ax=ax2, color=['orange', 'blue'], alpha=0.8)
ax2.set_title('대륙별 호텔 타입별 예약 수', fontsize=14, fontweight='bold')
ax2.set_xlabel('대륙')
ax2.set_ylabel('예약 수')
ax2.legend(['City Hotel', 'Resort Hotel'])
ax2.tick_params(axis='x', rotation=45)

# 3. 대륙별 취소율 계산
continent_cancel_rate = hotel_data.groupby('continent').apply(
    lambda x: (x['is_canceled'] == 1).sum() / len(x) * 100
).sort_values(ascending=False)

ax3 = axes[1, 0]
continent_cancel_rate.plot(kind='bar', ax=ax3, color='red', alpha=0.7)
ax3.set_title('대륙별 취소율', fontsize=14, fontweight='bold')
ax3.set_xlabel('대륙')
ax3.set_ylabel('취소율 (%)')
ax3.tick_params(axis='x', rotation=45)

# 4. 상위 대륙들의 월별 예약 패턴
ax4 = axes[1, 1]
top_continents = continent_counts.head(3).index
top_continent_data = continent_data[continent_data['continent'].isin(top_continents)]

month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

for continent in top_continents:
    continent_monthly = top_continent_data[top_continent_data['continent'] == continent].groupby('arrival_date_month').size()
    continent_monthly = continent_monthly.reindex(month_order, fill_value=0)
    ax4.plot(continent_monthly.index, continent_monthly.values, marker='o', label=continent, linewidth=2)

ax4.set_title('상위 3개 대륙의 월별 예약 패턴', fontsize=14, fontweight='bold')
ax4.set_xlabel('월')
ax4.set_ylabel('예약 수')
ax4.legend()
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

## 4.4.2 주요 국가별 예약 분포 (취소 제외)
plt.figure(figsize=(15, 6))

# 취소되지 않은 예약 중 상위 10개 국가만 추출
top_countries = hotel_data[hotel_data["is_canceled"] == 0]["country"].value_counts().nlargest(10).index
country_data = hotel_data[hotel_data["is_canceled"] == 0]
country_data = country_data[country_data["country"].isin(top_countries)]

# 국가별 예약 수
sns.countplot(data=country_data, x="country", hue="hotel", order=top_countries, palette="viridis")
plt.title("주요 10개 국가별 예약 분포 (취소 제외)", fontsize=16, fontweight='bold')
plt.xlabel("국가 코드")
plt.ylabel("예약 수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print(f"분석: 가장 많은 예약을 하는 국가는 {top_countries[0]} (Portugal) 입니다. 이어서 {top_countries[1]} (Great Britain) 순으로 나타납니다.")


## 4.5 평균 리드 타임 (Lead Time) 분석
plt.figure(figsize=(8, 6))
sns.boxplot(x='hotel', y='lead_time', data=hotel_data, palette=['darkorange', 'lightskyblue'])
plt.title('호텔 타입별 Lead Time 분포 (중앙값)', fontsize=16, fontweight='bold')
plt.xlabel('호텔 타입')
plt.ylabel('리드 타임 (일)')
plt.ylim(0, 400) # 이상치로 인해 범위 제한
plt.tight_layout()
plt.show()
print("분석: City Hotel이 Resort Hotel보다 예약 시점까지의 리드 타임이 더 짧은 경향을 보입니다. (즉, 더 임박해서 예약)")