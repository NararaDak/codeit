import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# 💡 배치 학습을 위해 필요한 유틸리티 모듈 추가
from torch.utils.data import TensorDataset, DataLoader 


# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'# Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False# 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')

def Lines():
    print("─" * 120)

# ----------------------------------------------------------------------
# [데이터 로드, 전처리, 분할 코드 (생략)]
# ----------------------------------------------------------------------
# (편의상 전처리 및 데이터 분할 코드는 그대로 유지하고 하단부만 재구성합니다.)
# ...
bank_data = pd.read_csv('D:/01.project/코드잇/미션/미션4/data/bank-additional-full.csv' ,sep = ';')
bank_data_test = pd.read_csv('D:/01.project/코드잇/미션/미션4/data/bank-additional.csv' ,sep = ';')
# ... [clean_dict, min_max_clean, age_by_5year_bands, preprocess_data 함수 정의 및 실행] ...
# ... [Feature 정의 및 StandardScaler 적용 코드 (생략)] ...



Lines()
print(bank_data.head())
Lines()
print(bank_data.info()) # 데이터 정보 확인
Lines()
# 결측치 확인 : 결측치는 없는 것으로 나옴.
null_df = bank_data.isnull().sum().reset_index()
null_df.columns = ['컬럼명', '결측치 개수']
print(null_df)
Lines()
#─────────────────────────────────────────
# 전역 매핑 저장용 딕셔너리
#─────────────────────────────────────────
clean_dict = {}  # 전역 매핑 저장용 딕셔너리

#─────────────────────────────────────────
# Min/Max 기준 균등 너비 범주화 함수
#─────────────────────────────────────────
min_max_dict = {}

def min_max_clean(df, col_name, range_count=10, add_name="_clean"):
    """
    mode='fit': 훈련 데이터의 min/max를 계산하고 딕셔너리에 저장합니다.
    mode='transform': 딕셔너리에 저장된 min/max를 사용하여 데이터를 변환합니다.
    """
    new_col_name = col_name + "_" + str(range_count) + add_name
     # 1. 스케일러 준비 및 모드 체크
    if(new_col_name not in min_max_dict):
        # fit 모드: MinMaxScaler를 새로 생성하고 현재 데이터로 학습
        scaler = MinMaxScaler()
        scaler.fit(df[[col_name]])
        min_max_dict[new_col_name] = scaler  # 학습된 스케일러 객체를 딕셔너리에 저장
        print(f"✅ '{col_name}'의 min/max가 학습되어 저장되었습니다.")
    else:
        # transform 모드: 저장된 스케일러를 사용
        scaler = min_max_dict[new_col_name] # 저장된 스케일러 객체를 사용
    # 2. 저장된 min/max 값과 bin 너비 계산 (fit 모드에서 이미 학습되었거나, transform 모드에서 로드된 값)
    min_val = scaler.data_min_[0]
    max_val = scaler.data_max_[0]
    bin_width = (max_val - min_val) / range_count
    # max_val + 1e-5: 경계값 오류를 방지하기 위해 최대값에 작은 값을 더함
    bins = [min_val + i * bin_width for i in range(range_count)] + [max_val + 1e-5]

    # 3. pd.cut을 사용하여 범주화 (fit/transform 모두 동일한 bins 사용)
    cut_series = pd.cut( 
        df[col_name], 
        bins=bins, 
        include_lowest=True,
        duplicates='drop'
    )
    df[new_col_name] = cut_series.cat.codes
    return df

#─────────────────────────────────────────
# 나이 범주화 함수
#─────────────────────────────────────────
def age_by_5year_bands(df: pd.DataFrame, source_col: str, target_col: str):
    # Age 컬럼을 11세 이하 (0), 이후 5세 단위로 범주화하고 90세 이상을 하나로 묶는 함수.
    # 구간 경계 설정
    bins = [1, 10] + list(range(15, 95, 5)) + [np.inf]
    
    # 각 구간의 레이블 (0,1,2,...)
    labels = list(range(len(bins) - 1))
    
    # 범주화 수행
    df[target_col] = pd.cut(
        df[source_col],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    ).astype(int)
    
    # 구간 문자열 추출
    intervals = pd.cut(
        df[source_col],
        bins=bins,
        right=True,
        include_lowest=True
    ).cat.categories.astype(str)
    
    # clean_dict에 (숫자 → 구간) 형태로 저장
    clean_dict[target_col] = {i: interval for i, interval in enumerate(intervals)}
    return df

#─────────────────────────────────────────
# 데이터 전처리 함수 
#─────────────────────────────────────────
def preprocess_data(df):
    #───────────────────────────────────────────────────────────────────────────────────────
    # ▣ 동일행 삭제 ▣ 
    #───────────────────────────────────────────────────────────────────────────────────────
    df =  df.drop_duplicates()
    #───────────────────────────────────────────────────────────────────────────────────────
    # ▣ pdays 999 특이값 처리 로직 추가 ▣ 
    #───────────────────────────────────────────────────────────────────────────────────────
    # 1. '이전 접촉 여부' 플래그 생성: 999가 아니면 1 (접촉했음), 999면 0 (접촉 안 했음)
    df['pdays_contacted'] = np.where(df['pdays'] == 999, 0, 1)
    # 2. 실제 경과 일수 컬럼 생성: 999를 NaN으로 대체
    df['pdays_actual'] = df['pdays'].replace(999, np.nan)
    # 3. pdays_actual의 NaN을 유효값의 중앙값으로 대체
    # (주의: .median()은 NaN을 제외하고 계산합니다)
    median_pdays = df['pdays_actual'].median()
    df['pdays_actual'] = df['pdays_actual'].fillna(median_pdays)
    # 4. 새로 생성된 이진 컬럼을 범주형으로 인코딩하여 clean_dict에 등록
    # 이진 컬럼이므로 그대로 0, 1로 사용하거나, 범주화하여 clean_dict에 등록할 수 있습니다.
    # 여기서는 범주화하여 clean_dict에 매핑을 남깁니다.
    df['pdays_contacted_clean'] = df['pdays_contacted'].astype('category').cat.codes
    clean_dict['pdays_contacted_clean'] = {0: 'No Previous Contact (999)', 1: 'Had Previous Contact'}
    # 5. 최종 연속형 변수인 pdays_actual을 min_max_clean을 통해 범주화
    min_max_clean(df, 'pdays_actual', range_count=10) # 5개 구간으로 범주화 예시
    #───────────────────────────────────────────────────────────────────────────────────────
    # ▣ contact_freq_ratio 생성 및 처리 ▣ 
    #───────────────────────────────────────────────────────────────────────────────────────
    # previous / campaign 비율 계산. campaign은 최소 1이므로 0으로 나누는 문제 없음.
    df['contact_freq_ratio'] = df['previous'] / df['campaign']
    
    # 생성된 비율 변수를 min_max_clean을 통해 범주화 (20개 구간)
    min_max_clean(df, 'contact_freq_ratio', range_count=20)
    #───────────────────────────────────────────────────────────────────────────────────────
    # ▣ recent_contact_flag 생성 및 인코딩 ▣ 
    #───────────────────────────────────────────────────────────────────────────────────────
    # 조건에 따른 값 할당
    conditions = [
        # 조건 1: 이전 접촉 없음 (pdays=999)
        (df['pdays'] == 999), 
        # 조건 2: 이전 접촉 성공
        (df['pdays'] != 999) & (df['poutcome'] == 'success'),
        # 조건 3: 이전 접촉 실패
        (df['pdays'] != 999) & (df['poutcome'] == 'failure')
    ]
    
    # 할당할 레이블
    choices = ['NoContact', 'Success', 'Failure']
    
    # np.select를 사용하여 새로운 범주형 컬럼 생성
    df['recent_contact_flag'] = np.select(conditions, choices, default='Other') # 'Other'는 발생하지 않음
    
    # 범주형으로 인코딩하여 clean_dict에 등록
    # 순서: NoContact(0), Failure(1), Success(2) (알파벳 순서대로 인코딩)
    df['recent_contact_flag_clean'] = df['recent_contact_flag'].astype('category').cat.codes
    
    # clean_dict에 매핑 저장
    clean_dict['recent_contact_flag_clean'] = dict(enumerate(df['recent_contact_flag'].astype('category').cat.categories))
    #───────────────────────────────────────────────────────────────────────────────────────
    # ▣ 카테고리 범주 숫자형 인코딩 ▣ 
    #───────────────────────────────────────────────────────────────────────────────────────
    categorical_cols = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 
        'contact', 'month', 'day_of_week', 'poutcome'
    ]

    # 범주형 변수 인코딩
    for col in categorical_cols:
        if col == 'month':
            # 'month' 특수 처리: 1월부터 12월 순서대로 인코딩 (Jan=0, Feb=1, ..., Dec=11)
            month_order = [
                'jan', 'feb', 'mar', 'apr', 'lender', # lender is a common value in bank marketing datasets,
                'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            ]
            # Lowercase all values for safety before setting categories
            df[col] = df[col].str.lower()
            
            # Ensure only relevant months are in the order, handling 'lender' if present,
            # or filtering to just the 12 calendar months if 'lender' isn't expected.
            present_months = df[col].unique().tolist()
            
            # Filter the month_order to only include months present in the data
            ordered_categories = [m for m in month_order if m in present_months]

            # Set the categorical type with explicit order
            df[col] = pd.Categorical(df[col], categories=ordered_categories, ordered=True)
            df[col + '_clean'] = df[col].cat.codes
            
            # 숫자 → 문자열 형태로 clean_dict 저장
            # cat.categories will now be in the specified order (jan, feb, ...)
            clean_dict[col + '_clean'] = dict(enumerate(df[col].cat.categories))
        elif col == 'day_of_week':
            # 'day_of_week' 특수 처리: 월요일부터 일요일 순서대로 인코딩 (mon=0, tue=1, ..., sun=6)
            day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
            df[col] = df[col].str.lower()
            df[col] = pd.Categorical(df[col], categories=day_order, ordered=True)
            df[col + '_clean'] = df[col].cat.codes
            # 숫자 → 문자열 형태로 clean_dict 저장
            clean_dict[col + '_clean'] = dict(enumerate(df[col].cat.categories))
        else:
            # 기본 범주형 변수 인코딩
            df[col] = df[col].astype('category')
            df[col + '_clean'] = df[col].cat.codes
            
            # 숫자 → 문자열 형태로 clean_dict 저장
            clean_dict[col + '_clean'] = dict(enumerate(df[col].cat.categories))

    # 타겟 변수 변환 및 clean_dict 등록
    target_map = {'no': 0, 'yes': 1}
    # Ensure target variable 'y' is consistent (e.g., lowercase)
    # Assuming 'y' is in the format 'yes'/'no'
    if 'y' in df.columns:
        df['y_clean'] = df['y'].astype(str).str.lower().map(target_map)
        clean_dict['y_clean'] = {v: k for k, v in target_map.items()} # {0:'no', 1:'yes'}
    else:
        # Handle case where 'y' might be missing in the input (e.g., prediction phase)
        pass 
    #------------------------------------------------------------------
    # 'job'별 평균 나이 계산
    job_age_mean = df.groupby('job')['age'].transform('mean')
    #  개인 나이와 직업 평균 나이의 차이 계산
    df['job_age_mean_diff'] = df['age'] - job_age_mean
    # 새로운 파생 변수에 대해 min_max_clean 적용 (범주화)
    # 20개 구간으로 나눠서 디테일한 차이를 반영
    min_max_clean(df, 'job_age_mean_diff', range_count=20)
    #------------------------------------------------------------------
    # 나이 범주화 (clean_dict에 자동 등록됨)
    age_by_5year_bands(df, 'age', 'age_clean')
    #min_max_clean(df, 'duration')
    #min_max_clean(df, 'campaign')
    min_max_clean(df, 'duration', 100)
    #min_max_clean(df, 'campaign', 100)
    min_max_clean(df, 'euribor3m', 100)
    min_max_clean(df, 'nr.employed', 10)
    min_max_clean(df, 'emp.var.rate',20)
    min_max_clean(df, 'cons.price.idx',20)
    min_max_clean(df, 'cons.conf.idx',20)
    return df

#─────────────────────────────────────────
# 데이터 전처리 실행 
#─────────────────────────────────────────
bank_data = preprocess_data(bank_data)
#─────────────────────────────────────────
bank_data_test = preprocess_data(bank_data_test)
#─────────────────────────────────────────
# StandardScaler  
#─────────────────────────────────────────


# 데이터 분할
# x_data, y_data 정의 (전처리된 bank_data 사용)
feature = [
"age_clean", "job_clean", "marital_clean", "education_clean", "default_clean", 
"housing_clean", "loan_clean", "contact_clean", "month_clean", "day_of_week_clean", 
"duration_100_clean", "campaign", "previous", "poutcome_clean", 
"cons.price.idx_20_clean", "cons.conf.idx_20_clean", "euribor3m_100_clean", 
"nr.employed_10_clean", "emp.var.rate_20_clean", "pdays_contacted", 
"pdays_actual_10_clean",
]
label = ['y_clean']

# StandardScaler 적용 (fit_transform은 이미 수행되었다고 가정)
# bank_data[feature] = std.fit_transform(bank_data[feature]) ...

x_data = bank_data[feature]
y_data = bank_data[label]
x_train,x_val,y_train,y_val = train_test_split(x_data,y_data,test_size=0.2,random_state=42)
x_test = bank_data_test[feature]
y_test = bank_data_test[label]

# 모델 학습
# ─────────────────────────────────────────
# 2. 하이퍼 파라메터 설정
# ─────────────────────────────────────────
LEARNING_RATE = 0.01
EPOCHS = 10000
H1 = len(feature)
H2 = H1
FEATURE_COUNT = H1
BATCH_SIZE = 500
#─────────────────────────────────────────
# 3. 모델 정의 (은닉층 1개, ReLU 활성화, 출력층 Sigmoid)
# ─────────────────────────────────────────
def makeModel():
    model = nn.Sequential(
        nn.Linear(len(feature), H1),
        nn.ReLU(),
        nn.Linear(H1, H2),
        nn.ReLU(),
        nn.Linear(H2, 1),
        nn.Sigmoid()
    )
    return model

model = makeModel()

# ─────────────────────────────────────────
# 4. 학습 루프 준비
# ─────────────────────────────────────────
# 2. 텐서를 생성하고 장치로 이동 (BCELoss를 위해 y_tensor는 (N, 1) 형태)
X_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# 3. TensorDataset 및 DataLoader 생성 💡 (배치 학습의 핵심)
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. 학습 루프 (배치 학습)
def train():
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0.0
        
        # 💡 DataLoader를 순회하며 미니 배치 학습 수행
        for X_batch, y_batch in train_loader:
            # X_batch와 y_batch는 이미 device로 이동되어 있음

            
            # 순전파
            outputs = model(X_batch)
            
            # 손실계산
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad() # 각 배치마다 기울기 초기화!
            # 역전파 및 최적화
            loss.backward() # 역전파를 통해 기울기 계산.
            optimizer.step() # 가중치 업데이트.
            
            # 배치 손실 누적 (정확한 에폭 평균 손실 계산을 위해 배치 크기 반영)
            total_epoch_loss += loss.item() * X_batch.size(0)

        # 에폭 평균 손실 계산
        avg_epoch_loss = total_epoch_loss / len(train_dataset)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:05d} | Avg. Loss: {avg_epoch_loss:.6f}")
            
        if avg_epoch_loss <= 0.00001:
            print(f"학습 종료: Epoch {epoch:05d} | Avg. Loss: {avg_epoch_loss:.6f}")
            break

# 학습 실행
train()

# 평가
from sklearn.metrics import accuracy_score
model.eval()

with torch.no_grad():
    # 평가 시 전체 X_tensor를 사용합니다. (이미 device에 있음)
    outputs = model(X_tensor)
    
    # 결과를 CPU로 이동하여 NumPy로 변환
    
    predicted = (outputs.numpy() > 0.5).astype(float)
    # y_train은 데이터프레임이 아닌 텐서의 CPU 버전(y_train_cpu)을 사용해야 합니다.
    accuracy = accuracy_score(y_train, predicted) 
    print(f"훈련 세트 정확도: {accuracy:.4f}")