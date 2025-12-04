import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform


# ----------------- 페이지 설정 (브라우저 탭 제목 등) -----------------
st.set_page_config(
    page_title="데이터 시각화 데모",  # 브라우저 탭 제목
    page_icon="📊",                   # 브라우저 탭 아이콘
    layout="wide",                    # 레이아웃 (wide 또는 centered)
    initial_sidebar_state="expanded"  # 사이드바 초기 상태
)

# ----------------- 한글 폰트 설정 -----------------
# Matplotlib에서 한글을 제대로 표시하기 위한 설정
plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# ----------------- 1. 앱 제목 설정 -----------------
st.title("간단한 Streamlit 데이터 시각화 데모")

# ----------------- 2. 사용자 입력 받기 (사이드바) -----------------
st.sidebar.header("설정 메뉴")

# 사용자로부터 데이터 포인트 개수를 입력받음
num_points = st.sidebar.slider(
    '데이터 포인트 개수',  # 슬라이더 라벨
    min_value=10,        # 최소값
    max_value=100,       # 최대값
    value=50,            # 초기값
    step=1               # 스텝
)
st.sidebar.write(f"선택한 데이터 포인트 개수: {num_points}")

# ----------------- 3. 데이터 생성 및 처리 -----------------

# num_points 개수만큼 랜덤 데이터를 생성
np.random.seed(42)  # 재현성을 위해 시드 설정
data = pd.DataFrame({
    'x': np.arange(num_points),
    # 누적합을 이용하여 약간의 경향성이 있는 랜덤 데이터 생성
    'y': np.cumsum(np.random.randn(num_points)) 
})

st.subheader(f"생성된 데이터 ({num_points}개 포인트)")

print(data.head())  # 콘솔에 데이터 출력 (디버깅용)

# ----------------- 4. 데이터 표시 및 시각화 -----------------

# (1) 데이터프레임 표시
st.dataframe(data.head()) # 상위 5개 행만 표시

# (2) Matplotlib을 사용한 시각화
fig, ax = plt.subplots()
ax.plot(data['x'], data['y'], label='누적 랜덤 값')
ax.set_title("누적 랜덤 데이터 변화")
ax.set_xlabel("인덱스")
ax.set_ylabel("값")
ax.grid(True)
ax.legend()

# Streamlit에 Matplotlib 그림을 표시
st.pyplot(fig)

# (3) Streamlit의 기본 차트 요소 사용
st.line_chart(data.set_index('x'))

# & 'd:\01.project\CodeIt\.venv\Scripts\python.exe' -m streamlit run "d:\01.project\CodeIt\streamit\simple_app.py"
