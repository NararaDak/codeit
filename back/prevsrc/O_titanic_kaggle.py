import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import tkinter as tk
from sklearn.model_selection import train_test_split
from tkinter import scrolledtext
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
import subprocess
import inspect
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from itertools import product

subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)

def Lines():
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    print(f"\n====================================[{filename}:{lineno}]====================================")


def MyShow():
    plt.show(block=False)
    plt.pause(3)
    plt.close()


script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'titanic_train.csv')
tt_train = pd.read_csv(data_path)
# print(tt_train.head())


data_path = os.path.join(script_dir, 'data', 'titanic_test.csv')
tt_test = pd.read_csv(data_path)
# print(tt_test.head())


xFeatures = ["Pclass","Initial","Cabin","Sex","Age","SibSp","Parch","FamilySize","Fare","Embarked","Solo"]
###################################
# Data Preprocessing
###################################
def preprocess_data(df):
    #--------------------------------------------------------------------------------------- 
    # Fare
    df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"), inplace=True)
    df['Fare'] = pd.qcut(df['Fare'], 5)
    df['Fare'] = df['Fare'].astype('category').cat.codes 
    #--------------------------------------------------------------------------------------- 
    # Embarked
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Embarked"] = df["Embarked"].map({"S":0,"C":1,"Q":2})
    #--------------------------------------------------------------------------------------- 
    # Sex
    df["Sex"] = df["Sex"].map({"male":0,"female":1})
    #--------------------------------------------------------------------------------------- 
    # --- 2. Initial (호칭) 특성 생성 및 숫자 변환 ---
    # 호칭 추출
    df['Initial'] = df.Name.str.extract('([A-Za-z]+)\.') 
    # 호칭 통합 및 치환
    df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                          ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
    
    # 'Dona' (테스트 데이터에 있음)와 같이 치환 목록에 없지만 추출된 문자열은 'Other'로 통합
    # 또한, 이 시점에서 혹시 남아있을 수 있는 NaN 값('NONE')을 미리 처리
    
    # 나머지 문자열 타이틀(예: 'Master', 'Dona')은 'Other'로 그룹핑하여 NaN 방지
    # 'Master', 'Mr', 'Mrs', 'Miss', 'Other', 'NONE' 만 남김
    all_known_titles = ['Master', 'Mr', 'Mrs', 'Miss', 'Other']
    
    # 치환 목록에 포함되지 않은 나머지 모든 타이틀을 'Other'로 강제 변환
    # (예: tt_test에 있는 'Dona'와 같은 타이틀을 'Other'로 처리)
    for title in df['Initial'].unique():
        if title not in all_known_titles:
            df['Initial'].replace(title, 'Other', inplace=True)
    # 최종적으로 NaN 값은 'NONE'으로 처리 (이 시점에서는 Name이 없는 경우만 해당)
    df['Initial'] = df['Initial'].fillna('NONE') 
    # Initial 숫자형 매핑
    initial_mapping = {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Other': 4, 'NONE': 5}
    df['Initial'] = df['Initial'].map(initial_mapping)
    #--------------------------------------------------------------------------------------- 
    # Age
    df['Age'].fillna(df.groupby('Initial')['Age'].transform('median'), inplace=True)
    # l1, l2 = [1,2,3], [0,1] # 'Sex'가 0과 1로 매핑되었다고 가정
    # for c,s in product(l1,l2):
        # msk = (df["Pclass"]==c) & (df["Sex"]==s)
        # df.loc[msk,"Age"] = df.loc[msk,"Age"].fillna(df.loc[msk,"Age"].median())
    #--------------------------------------------------------------------------------------- 
    # --- 3. Cabin 특성 생성 및 숫자 변환 ---
    df["Cabin"].fillna("N", inplace=True) 
    df['Cabin'] = df['Cabin'].str[0]
    # Cabin 숫자형 매핑
    cabin_mapping = {"A": 0.0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8,"N":3.0}
    df['Cabin'] = df['Cabin'].map(cabin_mapping)
    #--------------------------------------------------------------------------------------- 
    # --- 4. FamilySize 처리 ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    #--------------------------------------------------------------------------------------- 
    df['Solo'] = (df['FamilySize'] == 1)
    #--------------------------------------------------------------------------------------- 
    return df[xFeatures]
""" 
    # 구간화
    df['Age'] = pd.cut(df['Age'], bins=[0, 15, 25, 35, 60, 100], labels=[0, 1, 2, 3, 4], right=False)
    # 0: 아동 (0~14), 1: 청소년/청년 (15~24), 2: 장년 (25~34), 3: 중년 (35~59), 4: 노년 (60~)
    df['Age'] = df['Age'].astype(int) # 정수형으로 변환

    # 5-2. FamilySize 구간화 및 숫자 매핑
    # 1: 혼자, 2-4: 소가족, 5이상: 대가족/Other 로 나누는 예시
    df.loc[df['FamilySize'] == 1, 'FamilySize'] = 0  # 혼자 (Solo)
    df.loc[(df['FamilySize'] >= 2) & (df['FamilySize'] <= 4), 'FamilySize'] = 1  # 소/중 가족 (Small/Medium)
    df.loc[df['FamilySize'] >= 5, 'FamilySize'] = 2  # 대가족 (Large)
    
    # 5-3. Fare 구간화 및 숫자 매핑
    # pd.qcut을 사용하여 데이터의 분포에 따라 4개의 분위수(Quantile)로 나누는 예시
    # (qcut은 각 구간에 비슷한 수의 데이터가 들어가도록 함)
    df['Fare'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3])
    df['Fare'] = df['Fare'].astype(int) # 정수형으로 변환
 """
###################################

x = preprocess_data(tt_train)
y = tt_train["Survived"]

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=42)

# Lines() 함수와 x_train, y_train, x_val, y_val은 이미 정의되어 있다고 가정합니다.
# def Lines():
#     print("-" * 30)


def Logistic_fn(max_iter=200):
    model = LogisticRegression(max_iter =max_iter )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_val)
    acc = accuracy_score(y_val, y_predict)
    Lines()
    print(f"LogisticRegression : {acc}")
    Lines()
    print(classification_report(y_val, y_predict))
    return acc,model



def RandomForest_fn(n_estimators=100, max_depth=None, max_features='sqrt'):
    """
    RandomForestClassifier를 실행하는 함수.
    디폴트 값은 sklearn의 기본 설정을 따릅니다.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42 # 재현성을 위해 추가
    )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_val)
    acc = accuracy_score(y_val, y_predict)
    Lines()
    print(f"RandomForestClassifier Accuracy (n_estimators={n_estimators}, max_depth={max_depth}, max_features='{max_features}'): {acc}")
    Lines()
    print(classification_report(y_val, y_predict))
    return acc,model

# -----------------------------------------------------------------------------

def GradientBoosting_fn(n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    GradientBoostingClassifier를 실행하는 함수.
    디폴트 값은 sklearn의 기본 설정을 따릅니다.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42 # 재현성을 위해 추가
    )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_val)
    acc = accuracy_score(y_val, y_predict)
    Lines()
    print(f"GradientBoostingClassifier Accuracy (n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}): {acc}")
    Lines()
    print(classification_report(y_val, y_predict))
    return acc,model

# -----------------------------------------------------------------------------

def XGB_fn(n_estimators=100, learning_rate=0.3, max_depth=6):
    """
    XGBClassifier를 실행하는 함수.
    디폴트 값은 XGBoost의 기본 설정을 따릅니다.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric='logloss', # 기본값 경고 방지 및 분류 문제에 적합한 메트릭 지정
        use_label_encoder=False, # 향후 버전에서 제거될 경고 방지
        random_state=42 # 재현성을 위해 추가
    )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_val)
    acc = accuracy_score(y_val, y_predict)
    Lines()
    print(f"XGBClassifier Accuracy (n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}): {acc}")
    Lines()
    print(classification_report(y_val, y_predict))
    return acc,model

# -----------------------------------------------------------------------------
# 1. 서포트 벡터 머신 (SVM) 함수 추가
# -----------------------------------------------------------------------------
def SupportVectorMachine_fn(C=1.0, kernel='rbf', gamma='scale'):
    """
    Support Vector Machine (SVC) 분류기를 실행하는 함수.
    """
    # SVC는 데이터 스케일링에 민감하지만, 여기서는 raw 데이터를 사용합니다.
    model = SVC(
        C=C, 
        kernel=kernel, 
        gamma=gamma, 
        random_state=42, 
        probability=True # Stacking을 위해 확률 출력 설정
    )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_val)
    acc = accuracy_score(y_val, y_predict)
    Lines()
    print(f"SupportVectorMachine Accuracy (C={C}, kernel='{kernel}'): {acc}")
    Lines()
    print(classification_report(y_val, y_predict))
    return acc, model
# -----------------------------------------------------------------------------
# 2. 스태킹 앙상블 함수 추가
# -----------------------------------------------------------------------------
def StackingEnsemble_fn():
    """
    여러 기본 모델(Estimator)을 결합하는 Stacking 앙상블 함수.
    최종 모델(Final Estimator)로 로지스틱 회귀를 사용합니다.
    """
    # 기본 분류기 정의
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ('gbm', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        # SVM은 학습 시간이 길 수 있어 낮은 C 값을 사용하거나 제외할 수 있습니다.
        # ('svc', SVC(C=0.1, probability=True, random_state=42)) 
    ]

    # Stacking Classifier 정의 (meta-learner로 LogisticRegression 사용)
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=300, random_state=42),
        cv=5 # 교차 검증 횟수
    )
    
    # x_train, y_train은 전역 변수로 이미 준비됨
    model.fit(x_train, y_train)
    y_predict = model.predict(x_val)
    acc = accuracy_score(y_val, y_predict)
    Lines()
    print(f"StackingEnsemble Accuracy: {acc}")
    Lines()
    print(classification_report(y_val, y_predict))
    return acc, model
# -----------------------------------------------------------------------------
LogisticAcc,LogisticModel = Logistic_fn(max_iter=300)
RandomAcc,RandomModel = RandomForest_fn(n_estimators=300)
GradientAcc,GradientModel = GradientBoosting_fn(n_estimators=300,learning_rate = 0.01,max_depth=6)
XGBAcc,XGBModel = XGB_fn(n_estimators=300,learning_rate = 0.01,max_depth=6)
# 신규 모델
SVMAcc, SVMModel = SupportVectorMachine_fn(C=0.5, kernel='rbf') # C 값을 낮춰 학습 시간을 단축
StackingAcc, StackingModel = StackingEnsemble_fn()

# -----------------------------------------------------------------------------
Lines()
Lines()
print(f"Logistic: {LogisticAcc:.4f}")
print(f"Random: {RandomAcc:.4f}")
print(f"GradientAcc: {GradientAcc:.4f}")
print(f"XGBAcc: {XGBAcc:.4f}")
print(f"SVMAcc: {SVMAcc:.4f}")
print(f"StackingAcc: {StackingAcc:.4f}")
Lines()
Lines()
# -----------------------------------------------------------------------------
x_test = preprocess_data(tt_test)
def save_file(x,y,name):
    submission = pd.DataFrame({
        "PassengerId":x["PassengerId"],
        "Survived":y
    })
    submission.to_csv(f"d:/temp/Submission{name}.csv",index=False)
    print(f"Submission{name}.csv 파일이 생성 되었습니다.")

Lines()
y_predict = LogisticModel.predict(x_test)
save_file(tt_test,y_predict,"Logistic")

Lines()
y_predict = RandomModel.predict(x_test)
save_file(tt_test,y_predict,"Random")

Lines()
y_predict = GradientModel.predict(x_test)
save_file(tt_test,y_predict,"Gradient")

Lines()
y_predict = XGBModel.predict(x_test)
save_file(tt_test,y_predict,"XGB")

# 신규 모델 파일 저장
Lines()
y_predict = SVMModel.predict(x_test)
save_file(tt_test, y_predict, "SVM")

Lines()
y_predict = StackingModel.predict(x_test)
save_file(tt_test, y_predict, "Stacking")