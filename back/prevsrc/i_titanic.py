import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import tkinter as tk
from tkinter import scrolledtext
import lib.QS as qs


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

qs.Lines()
data = pd.read_csv('D:/01.project/코드잇/part-1-main/data/titanic.csv')
print(data.head())

qs.Lines()
#checking for total null values
print(data.isnull().sum())

f,ax=plt.subplots(1,2,figsize=(18,8))

# Survived 컬럼의 0, 1 값을 'Death', 'Survive'로 변환
survived_labels = data['Survived'].map({0: 'Death', 1: 'Survive'})

# 파이 차트 그리기
survived_labels.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

# 막대 차트 그리기
sns.countplot(x=survived_labels,ax=ax[1])
ax[1].set_title('Survived')
MyShow()


# 그룹별 생존률 막대 차트
data.groupby(['Sex','Survived'])['Survived'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot(x='Sex', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
MyShow()


# 객실 등급별 생존률 막대 차트
pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot(x='Pclass', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
MyShow()

# 성별과 객실 등급별 생존률 막대 차트
pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='summer_r')
sns.catplot(x='Pclass', y='Survived', hue='Sex', data=data, kind='point')
plt.show()

# 나이 통계
print('Oldest Passenger was of:',data['Age'].max(),'Years')
print('Youngest Passenger was of:',data['Age'].min(),'Years')
print('Average Age on the ship:',data['Age'].mean(),'Years')

# 객실 등급과 나이별 생존률 비교
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot(x= "Pclass",y="Age", hue="Survived", data=data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot(x="Sex",y="Age", hue="Survived", data=data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
MyShow()


# 이름에서 호칭 추출
data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
print(data.groupby('Initial')['Age'].mean() )#lets check the average age by Initials


# 나이 결측치 처리
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46
data.Age.isnull().any() #So no null values left finally 

# 나이 분포 비교
f,ax=plt.subplots(1,2,figsize=(20,10))
data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
MyShow()    

# 객실 등급과 호칭별 생존률 비교
sns.catplot(x='Pclass', y='Survived', col='Initial', data=data, kind='point')
plt.show()

