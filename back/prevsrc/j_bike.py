import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import tkinter as tk
from tkinter import scrolledtext

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')

import inspect
def Lines():
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    print(f"\n====================================[{filename}:{lineno}]====================================")


def MyShow():
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def show_popup(content, delay=3):
    """팝업 창으로 텍스트를 표시하고 지정된 시간 후 자동으로 닫는 함수
    
    Args:
        content: 표시할 내용
        delay (int): 창을 표시할 시간(초). 기본값은 3초
    """
    root = tk.Tk()
    root.title("메시지")
    root.geometry("600x400")
    
    # 스크롤 가능한 텍스트 위젯
    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # 내용 삽입
    text_widget.insert(tk.END, str(content))
    text_widget.config(state=tk.DISABLED)  # 읽기 전용으로 설정
    
    # 닫기 버튼
    close_button = tk.Button(root, text="닫기", command=root.destroy)
    close_button.pack(pady=5)
    
    # delay 시간 후 자동으로 창 닫기
    def auto_close():
        root.after(delay * 1000, root.destroy)  # delay 초 후 창 닫기
    
    # 자동 닫기 시작
    auto_close()
    root.mainloop()


bike = pd.read_csv('D:/01.project/코드잇/part-1-main/data/bike_sharing.csv')

# datetime 컬럼을 DatetimeIndex로 변환하고 인덱스로 설정
bike['datetime'] = pd.to_datetime(bike['datetime'])
bike.set_index('datetime', inplace=True)

# 월과 일 컬럼 추가
bike['month'] = bike.index.month
bike['day'] = bike.index.day
# 요일별 컬럼 추가
bike['weekday'] = bike.index.weekday
# 데이터 확인
show_popup(bike.head())


# 일별 평균 대여량 계산
daily_mean = bike['count'].resample('D').mean()

# 선 그래프 그리기
plt.figure(figsize=(12, 6))
sns.lineplot(x=daily_mean.index, y=daily_mean)
plt.title('일별 평균 대여량')
plt.xlabel('날짜')
plt.ylabel('평균 대여량')
plt.xticks(rotation=45)
plt.tight_layout()
MyShow()

# count 와  weather 컬럼의 관계
count_by_weather = bike.groupby('weather')['count'].mean()
plt.bar(count_by_weather.index, count_by_weather)
plt.title('weather 별 평균 대여량')
plt.xlabel('weather')
plt.ylabel('평균 대여량')
plt.xticks(rotation=45)
plt.tight_layout()
MyShow()

# weekday, weather, count 의 관계
sns.catplot(x='weekday', y='count', hue='weather', data=bike, kind='bar')
MyShow()


# season, weather, count 의 관계 
plt.figure(figsize=(12, 8))

# 막대 그래프 (배경)
sns.barplot(x='season', y='count', hue='weather', data=bike, alpha=0.7)
plt.title('Season별 Weather에 따른 대여량)')
plt.xlabel('Season')
plt.ylabel('대여량')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





