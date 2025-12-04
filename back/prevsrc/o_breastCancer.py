from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
def fn1():
    data = load_breast_cancer()

    # print(data.head())

    # x = data.drop(columns=["di","diagnosis"])
    # y = data["diagnosis"]

    x,y = data.data,data.target
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(x,y)

    plt.figure(figsize=(12,6))
    plot_tree(model,feature_names=data.feature_names,class_names=data.target_names,filled=True)
    plt.show()
#=======================================================================================
def fn2():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'breast-cancer.csv')
    data = pd.read_csv(data_path)
    print(data.head())

    x = data.drop(columns=["id","diagnosis"])
    y = data["diagnosis"]

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(x,y)

    plt.figure(figsize=(12,6))
    plot_tree(model,feature_names=x.columns.tolist(),class_names=["B","M"],filled=True)
    plt.show()


#=======================================================================================
from sklearn.model_selection import train_test_split
def fn3():
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # train_test_split은 4개의 결과(X_train, X_test, y_train, y_test)를 반환합니다.
    # 순서를 정확하게 맞춰주어야 합니다. (X_train,y_train,X_test,y_test -> X_train,X_test,y_train,y_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit 오류 해결: 이제 임포트가 완료되어 fit 메서드를 사용할 수 있습니다.
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    # DataFrame을 Series로 변환
    feature_importances = pd.Series(model.feature_importances_, index=data.feature_names)
    print(feature_importances.sort_values(ascending=False).head(10))




def fn4():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    model =GradientBoostingClassifier(learning_rate=0.1,n_estimators=100)
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)
    print("GradientBoostingClassifier Accuracy:",accuracy_score(y_test,y_predict))
    return accuracy_score(y_test,y_predict)
gradientAcc = fn4()


def fn5():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
    model =XGBClassifier(learning_rate=0.1,n_estimators=100)
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)
    print("XGBClassifier Accuracy:",accuracy_score(y_test,y_predict))
    return accuracy_score(y_test,y_predict)
xgbAcc = fn5()

print(f"GradientBoostingClassifier({gradientAcc}):XGBClassifier({xgbAcc})")
import sys

print(sys.executable)

# 현재 노트북이 사용하는 파이썬 환경의 pip을 사용해 설치합니다.
#{sys.executable} -m pip install xgboost