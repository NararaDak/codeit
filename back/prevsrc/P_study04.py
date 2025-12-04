import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 스케일러 임포트

# ================================
# 그래프 설정
# ================================
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')  # 그래프 스타일 설정
plt.rcParams["figure.figsize"] = 10, 6 # 기본 그래프 크기 설정


def impute_median(series):
    """
    결측값을 해당 시리즈의 중앙값으로 대체하는 함수
    
    Parameters:
    series (pandas.Series): 결측값이 포함된 데이터 시리즈
    
    Returns:
    pandas.Series: 중앙값으로 결측값이 채워진 시리즈
    """
    return series.fillna(series.median())

# ================================
# 데이터 로드
# ================================
# 현재 스크립트가 위치한 디렉토리 경로 가져오기
script_dir = os.path.dirname(os.path.abspath(__file__))
# 호텔 데이터 파일의 전체 경로 생성
data_path = os.path.join(script_dir, 'data', 'titanic.csv')
# CSV 파일을 DataFrame으로 읽어오기
tt_data = pd.read_csv(data_path)

# ================================
# Age Normalization/Standardization
# ================================
tt_age = tt_data["Age"].dropna()
print(tt_age)

# ================================
# Normalization/Standardization 함수 구현 (직접 구현과 라이브러리 사용 버전)
# ================================

def NormalizationFnNotLib(series):
    # 라이브러리 사용 없이 Min-Max 정규화를 수행하는 함수
    # 입력 시리즈를 [0, 1] 범위로 변환
    
    min_val = series.min()
    max_val = series.max()
    # 분모가 0인 경우, 변화가 없으므로 원본 시리즈 반환
    if max_val - min_val == 0:
        return series
    return (series - min_val) / (max_val - min_val)

def StandardFnNotLib(series):
    # 라이브러리 사용 없이 표준화를 수행하는 함수
    # 입력 시리즈를 평균 0, 표준편차 1로 변환
    
    mean_val = series.mean()
    std_val = series.std()
    # 표준편차가 0일 경우, 변화가 없으므로 원본 시리즈 반환
    if std_val == 0:
        return series
    return (series - mean_val) / std_val


def NormalizationFnLib(series):
    # scikit-learn의 MinMaxScaler를 사용하여 정규화를 수행하는 함수
    # 입력 시리즈를 [0, 1] 범위로 변환
    
    scaler = MinMaxScaler()
    reshaped = series.values.reshape(-1, 1)  # 스케일러를 위한 2차원 배열로 변환
    scaled = scaler.fit_transform(reshaped)
    return pd.Series(scaled.flatten(), index=series.index)


def StandardFnLib(series):
    # scikit-learn의 StandardScaler를 사용하여 표준화를 수행하는 함수
    # 입력 시리즈를 평균 0, 표준편차 1로 변환
    
    scaler = StandardScaler()
    reshaped = series.values.reshape(-1, 1)  # 스케일러를 위한 2차원 배열로 변환
    scaled = scaler.fit_transform(reshaped)
    return pd.Series(scaled.flatten(), index=series.index)
# ================================
# 함수 테스트 및 결과 출력
# ================================
normalized_manual = NormalizationFnNotLib(tt_age)
standardized_manual = StandardFnNotLib(tt_age)
normalized_lib = NormalizationFnLib(tt_age)
standardized_lib = StandardFnLib(tt_age)

print("\n수작업 정규화 결과:")
print(normalized_manual.head())
print("\n수작업 표준화 결과:")
print(standardized_manual.head())
print("\n라이브러리 사용 정규화 결과:")
print(normalized_lib.head())
print("\n라이브러리 사용 표준화 결과:")
print(standardized_lib.head())

# ================================
# sex one-hot 
# ================================
sex_onehot = pd.DataFrame({"Sex":tt_data["Sex"].dropna()})
encoded = pd.get_dummies(sex_onehot,columns=["Sex"])
print(sex_onehot.head())
print(encoded.head())

# ================================
# Pclass,Age,Fare -> Survived
# ================================
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


tt_org = tt_data[["Pclass","Age","Fare","Survived"]].dropna()
print(tt_org.shape)
x = tt_org[["Pclass","Fare","Survived"]]
y = tt_org[["Age"]]

# Ridge(L2)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
rg = Ridge(alpha=1.0)
rg.fit(x_train,y_train)
print("Ridge score",rg.score(x_test,y_test))

# Lasso(L1)
lss = Lasso(alpha=0.01)
lss.fit(x_train,y_train)
print("Lasso score",lss.score(x_test,y_test))
