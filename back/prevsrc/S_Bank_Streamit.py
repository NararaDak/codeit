import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_val_score # Streamlit에서 직접 사용하기엔 부적합

# streamlit run D:\01.project\코드잇\src\S_Bank_Streamit.py
# ----------------------------------------
# 원본 코드의 헬퍼 함수 및 전처리 로직
# (Lines, min_max_clean, age_by_5year_bands)
# ----------------------------------------

# 한글 폰트 설정 (Streamlit 클라우드 배포 시 별도 설정 필요)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('fivethirtyeight')

def Lines():
    # Streamlit에서는 st.divider()를 사용하는 것이 더 좋습니다.
    st.divider()

# 전역 매핑 저장용 딕셔너리
clean_dict = {}
min_max_dict = {}

def min_max_clean(df, col_name, range_count=10, add_name="_clean"):
    global min_max_dict
    new_col_name = col_name + "_" + str(range_count) + add_name
    if(new_col_name not in min_max_dict):
        scaler = MinMaxScaler()
        scaler.fit(df[[col_name]])
        min_max_dict[new_col_name] = scaler
        # st.write(f"✅ '{col_name}'의 min/max가 학습되어 저장되었습니다.") # Streamlit에서는 print 대신 st.write
    else:
        scaler = min_max_dict[new_col_name]
    
    min_val = scaler.data_min_[0]
    max_val = scaler.data_max_[0]
    bin_width = (max_val - min_val) / range_count
    bins = [min_val + i * bin_width for i in range(range_count)] + [max_val + 1e-5]

    cut_series = pd.cut( 
        df[col_name], 
        bins=bins, 
        include_lowest=True,
        duplicates='drop'
    )
    df[new_col_name] = cut_series.cat.codes
    return df

def age_by_5year_bands(df: pd.DataFrame, source_col: str, target_col: str):
    bins = [1, 10] + list(range(15, 95, 5)) + [np.inf]
    labels = list(range(len(bins) - 1))
    
    df[target_col] = pd.cut(
        df[source_col],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    ).astype(int)
    
    intervals = pd.cut(
        df[source_col],
        bins=bins,
        right=True,
        include_lowest=True
    ).cat.categories.astype(str)
    
    clean_dict[target_col] = {i: interval for i, interval in enumerate(intervals)}
    return df

# [수정됨] 'job'별 평균 나이를 저장하고 재사용하도록 수정
def preprocess_data(df):
    df = df.drop_duplicates()
    
    # ▣ pdays 999 특이값 처리 로직 추가 ▣
    df['pdays_contacted'] = np.where(df['pdays'] == 999, 0, 1)
    df['pdays_actual'] = df['pdays'].replace(999, np.nan)
    
    # 'pdays_actual_median'이 저장되어 있지 않으면 훈련 데이터에서 계산
    if 'pdays_actual_median' not in clean_dict:
        clean_dict['pdays_actual_median'] = df['pdays_actual'].median()
        
    median_pdays = clean_dict['pdays_actual_median']
    df['pdays_actual'] = df['pdays_actual'].fillna(median_pdays)
    
    df['pdays_contacted_clean'] = df['pdays_contacted'].astype('category').cat.codes
    if 'pdays_contacted_clean' not in clean_dict:
        clean_dict['pdays_contacted_clean'] = {0: 'No Previous Contact (999)', 1: 'Had Previous Contact'}
    
    min_max_clean(df, 'pdays_actual', range_count=10)
    
    # ▣ contact_freq_ratio 생성 및 처리 ▣
    df['contact_freq_ratio'] = df['previous'] / (df['campaign'] + 1e-6) # 0으로 나누기 방지
    min_max_clean(df, 'contact_freq_ratio', range_count=20)
    
    # ▣ recent_contact_flag 생성 및 인코딩 ▣
    conditions = [
        (df['pdays'] == 999), 
        (df['pdays'] != 999) & (df['poutcome'] == 'success'),
        (df['pdays'] != 999) & (df['poutcome'] == 'failure')
    ]
    choices = ['NoContact', 'Success', 'Failure']
    df['recent_contact_flag'] = np.select(conditions, choices, default='Other')
    df['recent_contact_flag_clean'] = df['recent_contact_flag'].astype('category').cat.codes
    if 'recent_contact_flag_clean' not in clean_dict:
        clean_dict['recent_contact_flag_clean'] = dict(enumerate(df['recent_contact_flag'].astype('category').cat.categories))

    # ▣ 카테고리 범주 숫자형 인코딩 ▣
    categorical_cols = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 
        'contact', 'month', 'day_of_week', 'poutcome'
    ]
    
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    day_order = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.lower() # 예측 시 입력될 단일 데이터도 처리 가능하게
        
        if col == 'month':
            present_months = df[col].unique().tolist()
            ordered_categories = [m for m in month_order if m in present_months]
            df[col] = pd.Categorical(df[col], categories=ordered_categories, ordered=True)
        elif col == 'day_of_week':
            df[col] = pd.Categorical(df[col], categories=day_order, ordered=True)
        else:
            df[col] = df[col].astype('category')
            
        df[col + '_clean'] = df[col].cat.codes
        
        if col + '_clean' not in clean_dict:
            clean_dict[col + '_clean'] = dict(enumerate(df[col].cat.categories))
            # 예측을 위해 (문자열 -> 숫자) 매핑도 저장
            clean_dict[col + '_map'] = {v: k for k, v in clean_dict[col + '_clean'].items()}


    if 'y' in df.columns:
        target_map = {'no': 0, 'yes': 1}
        df['y_clean'] = df['y'].astype(str).str.lower().map(target_map)
        if 'y_clean' not in clean_dict:
            clean_dict['y_clean'] = {v: k for k, v in target_map.items()}

    # ▣ [수정] job_age_mean 처리 ▣
    # 1. 훈련 시 'job_age_mean_map' 계산 및 저장
    if 'job_age_mean_map' not in clean_dict:
        clean_dict['job_age_mean_map'] = df.groupby('job')['age'].mean().to_dict()
        
    # 2. 저장된 맵을 사용하여 매핑 (훈련/테스트/예측 공통)
    # job_age_mean = df['job'].map(clean_dict['job_age_mean_map'])
    job_age_mean = df['job'].astype(str).map(clean_dict['job_age_mean_map'])

    
    # 3. 맵에 없는 새로운 job이 예측에 들어올 경우를 대비해
    #    전체 평균 나이로 맵의 NaN 값을 채움
    if 'age_mean_global' not in clean_dict:
         clean_dict['age_mean_global'] = df['age'].mean()
    
    job_age_mean = job_age_mean.fillna(clean_dict['age_mean_global'])
    
    df['job_age_mean_diff'] = df['age'] - job_age_mean
    min_max_clean(df, 'job_age_mean_diff', range_count=20)
    
    # 나이 및 기타 연속형 변수 범주화
    age_by_5year_bands(df, 'age', 'age_clean')
    min_max_clean(df, 'duration', 100)
    min_max_clean(df, 'euribor3m', 100)
    min_max_clean(df, 'nr.employed', 10)
    min_max_clean(df, 'emp.var.rate', 20)
    min_max_clean(df, 'cons.price.idx', 20)
    min_max_clean(df, 'cons.conf.idx', 20)
    
    return df

# ----------------------------------------
# 원본 코드의 모델 및 피처 정의
# ----------------------------------------
feature = [
    "age_clean", "job_clean", "marital_clean", "education_clean", "default_clean",
    "housing_clean", "loan_clean", "contact_clean", "month_clean", "day_of_week_clean",
    # "duration_100_clean",
      "campaign", "previous", "poutcome_clean",
    "cons.price.idx_20_clean", "cons.conf.idx_20_clean", "euribor3m_100_clean",
    "nr.employed_10_clean", "emp.var.rate_20_clean", "pdays_contacted",
    "pdays_actual_10_clean"
    # "contact_freq_ratio_20_clean", # 원본에서 주석 처리됨
    # "recent_contact_flag_clean",
    # "job_age_mean_diff_20_clean"
]
label = ['y_clean']

# 모델 정의
def makeModel(input_dim, h1, h2):
    model = nn.Sequential(
        nn.Linear(input_dim, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
    )
    return model

# ----------------------------------------
# Streamlit 캐싱을 활용한 함수 정의
# ----------------------------------------

# @st.cache_data: 데이터 로딩 및 전처처럼 결과가 바뀌지 않는 작업을 캐시
@st.cache_data
def load_and_preprocess_data():
    # 데이터 로드
    try:
        BASE_DIR = r"D:\01.project\CodeIt\data\bank"
        bank_data = pd.read_csv(BASE_DIR + '\\bank-additional-full.csv', sep=';')
        bank_data_test = pd.read_csv(BASE_DIR + '\\bank-additional.csv', sep=';')
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return None, None, None, None

    # 전처리 (중요: 훈련 데이터로 먼저 딕셔너리 채우기)
    bank_data = preprocess_data(bank_data)
    bank_data_test = preprocess_data(bank_data_test)

    # 데이터 분할 (원본 로직 존중)
    train_val_df = bank_data[feature + label].reset_index(drop=True)
    test_df = bank_data_test[feature + label].reset_index(drop=True)
    
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    
    return train_df, val_df, test_df, feature, label, clean_dict


# @st.cache_data: 스케일링 (데이터셋에 의존)
@st.cache_data
def scale_data(train_df, val_df, test_df, _feature_cols): # _feature_cols는 캐시 키로 사용
    scaler = StandardScaler()
    
    # 훈련 데이터로 fit
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[_feature_cols] = scaler.fit_transform(train_df[_feature_cols])
    
    # 검증 및 테스트 데이터는 transform
    val_df_scaled[_feature_cols] = scaler.transform(val_df[_feature_cols])
    test_df_scaled[_feature_cols] = scaler.transform(test_df[_feature_cols])
    
    return train_df_scaled, val_df_scaled, test_df_scaled, scaler

# @st.cache_resource: 모델, 스케일러 등 리소스를 캐시
# 하이퍼파라미터가 바뀔 때마다 재학습하도록 인자를 받음
# 텐서 인수를 제거하고 하이퍼파라미터만 남깁니다.
# 주의: 이 함수는 더 이상 @st.cache_resource로 사용할 수 없습니다.
# 모델을 캐시하려면 모델 저장/로드 로직을 사용하거나, 캐시를 포기해야 합니다.

# Streamlit 캐시를 사용할 수 없으므로, 캐시를 제거하고 함수 이름을 변경합니다.
def run_model_training(X_tensor, y_tensor, lr, epochs, h1, h2, feature_count): 
    model = makeModel(feature_count, h1, h2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    progress_bar = st.progress(0, text="모델 학습 중...")
    loss_list = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor) # X_tensor 사용
        loss = criterion(outputs, y_tensor) # y_tensor 사용
        # ... (나머지 학습 로직은 동일) ...
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        if epoch % 100 == 0 or epoch == epochs - 1:
            progress_bar.progress((epoch + 1) / epochs, text=f"Epoch {epoch:05d} | Loss: {loss.item():.6f}")

        if loss.item() <= 0.00001:
            progress_bar.progress(1.0, text=f"학습 조기 종료: Epoch {epoch:05d} | Loss: {loss.item():.6f}")
            break
            
    return model, loss_list

def evaluate_model(model, x_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(x_tensor)
        # BCEWithLogitsLoss를 사용했으므로 Sigmoid를 통과시켜 확률로 변환
        probs = torch.sigmoid(outputs)
        predicted = (probs.numpy() > 0.5).astype(float)
        accuracy = accuracy_score(y_tensor, predicted)
    return accuracy

# ----------------------------------------
# Streamlit 앱 UI 구성
# ----------------------------------------

st.set_page_config(layout="wide")
st.title("🏦 은행 마케팅 예측 모델 (PyTorch & Streamlit)")

# 1. 데이터 로드 및 전처리
data_load_state = st.text("데이터 로딩 및 전처리 중...")
data_tuple = load_and_preprocess_data()
if data_tuple[0] is None:
    st.stop()

train_df, val_df, test_df, feature_cols, label_col , mydict= data_tuple
data_load_state.text("데이터 로딩 및 전처리 완료!")
clean_dict = mydict

st.header("1. 데이터 탐색")
if st.checkbox("처리된 훈련 데이터 보기"):
    st.dataframe(train_df.head(10))
if st.checkbox("전처리 매핑 딕셔너리 보기 (clean_dict)"):
    st.json(clean_dict, expanded=False)

Lines()

# 2. 사이드바: 하이퍼파라미터 설정
st.sidebar.header("⚙️ 모델 하이퍼파라미터")
LEARNING_RATE = st.sidebar.slider("학습률 (Learning Rate)", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
EPOCHS = st.sidebar.slider("에포크 (Epochs)", 1000, 20000, 5000, 1000) # 기본 20000은 너무 긺
H1 = st.sidebar.number_input("은닉층 1 크기 (H1)", 1, 128, len(feature_cols))
H2 = st.sidebar.number_input("은닉층 2 크기 (H2)", 1, 128, len(feature_cols))

# 3. 모델 학습 및 평가
st.header("2. 모델 학습 및 평가")

if st.button("🚀 모델 학습 시작하기", type="primary"):
    
    with st.spinner("데이터 스케일링 및 텐서 변환 중..."):
        # 3-1. 스케일링 (데이터에 의존)
        train_scaled, val_scaled, test_scaled, scaler = scale_data(train_df, val_df, test_df, feature_cols)
        
        # 3-2. 텐서 변환
        X_train_tensor = torch.tensor(train_scaled[feature_cols].values, dtype=torch.float32)
        y_train_tensor = torch.tensor(train_scaled[label_col].values, dtype=torch.float32)
        X_val_tensor = torch.tensor(val_scaled[feature_cols].values, dtype=torch.float32)
        y_val_tensor = torch.tensor(val_scaled[label_col].values, dtype=torch.float32)
        X_test_tensor = torch.tensor(test_scaled[feature_cols].values, dtype=torch.float32)
        y_test_tensor = torch.tensor(test_scaled[label_col].values, dtype=torch.float32)

    with st.spinner("모델 학습 중... (시간이 걸릴 수 있습니다)"):
        # 3-3. 모델 학습 (하이퍼파라미터에 의존)
        model, loss_history = run_model_training(X_train_tensor, y_train_tensor, LEARNING_RATE, EPOCHS, H1, H2, len(feature_cols))
        st.success("모델 학습 완료!")

        # 3-4. 학습 곡선 시각화
        st.subheader("📊 학습 손실(Loss) 곡선")
        fig, ax = plt.subplots()
        ax.plot(loss_history)
        ax.set_title("Training Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCEWithLogitsLoss")
        st.pyplot(fig)

        # 3-5. 평가
        st.subheader("🎯 모델 정확도")
        val_accuracy = evaluate_model(model, X_val_tensor, y_val_tensor)
        test_accuracy = evaluate_model(model, X_test_tensor, y_test_tensor)
        
        col1, col2 = st.columns(2)
        col1.metric("검증 세트 정확도 (Validation Accuracy)", f"{val_accuracy:.4f}")
        col2.metric("테스트 세트 정확도 (Test Accuracy)", f"{test_accuracy:.4f}")

        # 3-6. 예측을 위해 학습된 모델과 스케일러를 세션 상태에 저장
        st.session_state['trained_model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['feature_cols'] = feature_cols

Lines()

# 4. 실시간 예측
st.header("3. 🧑‍💻 실시간 예측")

if ('trained_model' in st.session_state) and ('job_map' in clean_dict): # 이 조건 추가
    st.info("위에서 학습된 모델과 스케일러를 사용하여 예측합니다.")
    # 원본 데이터의 특성을 기반으로 입력 폼 생성
    # 전처리에 필요한 *모든* 원본 컬럼이 필요합니다.
    with st.form("prediction_form"):
        st.subheader("고객 정보 입력")
        
        # 입력 편의를 위해 2단, 3단 컬럼 사용
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("나이 (age)", min_value=17, max_value=100, value=40)
            job = st.selectbox("직업 (job)", options=clean_dict['job_map'].keys())
            marital = st.selectbox("결혼 여부 (marital)", options=clean_dict['marital_map'].keys())
            education = st.selectbox("교육 수준 (education)", options=clean_dict['education_map'].keys())
            default = st.selectbox("신용 불량 여부 (default)", options=clean_dict['default_map'].keys())

        with col2:
            housing = st.selectbox("주택 대출 (housing)", options=clean_dict['housing_map'].keys())
            loan = st.selectbox("개인 대출 (loan)", options=clean_dict['loan_map'].keys())
            contact = st.selectbox("연락 유형 (contact)", options=clean_dict['contact_map'].keys())
            month = st.selectbox("마지막 연락 월 (month)", options=clean_dict['month_map'].keys())
            day_of_week = st.selectbox("마지막 연락 요일 (day_of_week)", options=clean_dict['day_of_week_map'].keys())

        with col3:
            duration = st.number_input("마지막 연락 시간(초) (duration)", min_value=0, value=180)
            campaign = st.number_input("캠페인 연락 횟수 (campaign)", min_value=1, value=2)
            pdays = st.number_input("이전 캠페인 후 경과 일 (pdays)", min_value=0, value=999) # 999가 기본
            previous = st.number_input("이전 캠페인 연락 횟수 (previous)", min_value=0, value=0)
            poutcome = st.selectbox("이전 캠페인 결과 (poutcome)", options=clean_dict['poutcome_map'].keys())
            
        st.subheader("경제 지표 입력")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            emp_var_rate = st.number_input("고용 변동률 (emp.var.rate)", value=-0.1, format="%.1f")
        with c2:
            cons_price_idx = st.number_input("소비자 물가지수 (cons.price.idx)", value=93.2, format="%.1f")
        with c3:
            cons_conf_idx = st.number_input("소비자 신뢰지수 (cons.conf.idx)", value=-42.0, format="%.1f")
        with c4:
            euribor3m = st.number_input("유리보 3개월 (euribor3m)", value=1.313, format="%.3f")
        with c5:
            nr_employed = st.number_input("고용자 수 (nr.employed)", value=5099.1, format="%.1f")

        submitted = st.form_submit_button("예측하기")

    if submitted:
        # 1. 입력 데이터를 DataFrame으로 변환
        input_data = {
            'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
            'housing': housing, 'loan': loan, 'contact': contact, 'month': month, 'day_of_week': day_of_week,
            'duration': duration, 'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
            'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx, 'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m, 'nr.employed': nr_employed
        }
        input_df = pd.DataFrame([input_data])
        
        # 2. 훈련 데이터와 동일하게 전처리
        # (clean_dict, min_max_dict가 이미 채워져 있으므로 재사용됨)
        try:
            processed_input_df = preprocess_data(input_df)
            feature_input_df = processed_input_df[st.session_state['feature_cols']]
        
            # 🚨 디버깅 1: 스케일링 전 값 확인
            # st.dataframe(feature_input_df) # 디버깅 시 잠시 활성화
            
            # 3. 저장된 스케일러로 변환
            scaled_input_data = st.session_state['scaler'].transform(feature_input_df)
            
            # 🚨 디버깅 2: 스케일링 후 값 확인
            # st.dataframe(pd.DataFrame(scaled_input_data, columns=st.session_state['feature_cols'])) # 디버깅 시 잠시 활성화
            
            # 4. 텐서로 변환
            input_tensor = torch.tensor(scaled_input_data, dtype=torch.float32)
            
            # 5. 모델 예측
            model = st.session_state['trained_model']
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item() # 확률률
                prediction = "가입 (Yes)" if prob > 0.5 else "미가입 (No)"

            # 6. 결과 표시
            st.subheader("✨ 예측 결과")
            if prediction == "가입 (Yes)":
                st.success(f"**{prediction}** (가입 확률: {prob:.2%})")
            else:
                st.error(f"**{prediction}** (가입 확률: {prob:.2%})")
                
            with st.expander("모델 입력값 보기 (전처리 및 스케일링 완료)"):
                st.dataframe(pd.DataFrame(scaled_input_data, columns=st.session_state['feature_cols']))
                
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.error("입력값을 확인하거나 모델을 다시 학습시켜주세요.")

else:
    st.warning("먼저 '모델 학습 시작하기' 버튼을 눌러 모델을 학습시켜주세요.")

