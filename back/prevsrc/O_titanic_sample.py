import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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
import subprocess
import os

subprocess.run('cls' if os.name == 'nt' else 'clear', shell=True)

warnings.filterwarnings('ignore')
def Lines():
    print("-" * 60)

# data import
train = pd.read_csv('d:/temp/titanic_train.csv')
test = pd.read_csv('d:/temp/titanic_test.csv')
gender_submission = pd.read_csv('d:/temp/gender_submission.csv')


def preprocess_data(df):
    df["Sex_clean"] = df["Sex"].astype("category").cat.codes
    df["Embarked"].fillna("S",inplace=True)
    df["Embarked_clean"] =  df["Embarked"].astype("category").cat.codes 
    df["Family"] = 1 + df["SibSp"] + df["Parch"]
    df["Solo"] = (df["Family"] ==1)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["FareBin"] = pd.qcut(df["Fare"],5)
    df["FareBin"].value_counts()
    df["Fare_clean"] = df["FareBin"].astype("category").cat.codes
    df["Fare_clean"].value_counts()
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    df["Title_clean"] = df["Title"].astype("category").cat.codes
    df["Age"].fillna(df.groupby("Title")["Age"].transform("median"),inplace=True)
    df.loc[ df['Age'] <= 10, 'Age_clean'] = 0
    df.loc[(df['Age'] > 10) & (df['Age'] <= 16), 'Age_clean'] = 1
    df.loc[(df['Age'] > 16) & (df['Age'] <= 20), 'Age_clean'] = 2
    df.loc[(df['Age'] > 20) & (df['Age'] <= 26), 'Age_clean'] = 3
    df.loc[(df['Age'] > 26) & (df['Age'] <= 30), 'Age_clean'] = 4
    df.loc[(df['Age'] > 30) & (df['Age'] <= 36), 'Age_clean'] = 5
    df.loc[(df['Age'] > 36) & (df['Age'] <= 40), 'Age_clean'] = 6
    df.loc[(df['Age'] > 40) & (df['Age'] <= 46), 'Age_clean'] = 7
    df.loc[(df['Age'] > 46) & (df['Age'] <= 50), 'Age_clean'] = 8
    df.loc[(df['Age'] > 50) & (df['Age'] <= 60), 'Age_clean'] = 9
    df.loc[ df['Age'] > 60, 'Age_clean'] = 10
    cabin_map = { 'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7 }
    df["Cabin_clean"] = df["Cabin"].str[:1]
    df["Cabin_clean"] = df["Cabin_clean"].map(cabin_map)
    df["Cabin_clean"] = df["Cabin_clean"].fillna(df.groupby("Pclass")["Cabin_clean"].transform("median"))
    df['Age_clean'].fillna(df['Age_clean'].median(), inplace=True)
    df['Cabin_clean'].fillna(df['Cabin_clean'].median(), inplace=True)



feature = [
    'Pclass',
    'SibSp',
    'Parch',
    'Sex_clean',
    'Embarked_clean',
    'Family',
    'Solo',
    'Title_clean',
    'Age_clean',
    'Cabin_clean',
    'Fare_clean',
]
label = [
    'Survived',
]


preprocess_data(train)
preprocess_data(test)

data = train[feature]
target = train[label]

k_fold = KFold(n_splits=10,shuffle=True,random_state=0)

def kfold_score(name, model):
    Lines()
    accuracy = cross_val_score(model, data, target, cv=k_fold, scoring='accuracy', ).mean()
    print(f"{name}교차 검증 평균 정확도 (Accuracy): {accuracy:.4f}")
    Lines()
    model.fit(data,target)
    y_predict = model.predict(test[feature])
    submission = pd.DataFrame({
        "PassengerId":test["PassengerId"],
        "Survived":y_predict
    })
    submission.to_csv(f"d:/temp/Submission_{name}.csv",index=False)
    #print(f"Submission_{name}.csv 파일이 생성 되었습니다.")

model = RandomForestClassifier(n_estimators=50,max_depth=6,random_state=0)
kfold_score("Random",model)

model = LogisticRegression(max_iter =100 )
kfold_score("Logistic",model)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42) # 재현성을 위해 추가)
kfold_score("Gradient",model)

model = XGBClassifier(n_estimators=300, learning_rate= 0.01, max_depth=6, eval_metric='logloss', use_label_encoder=False, random_state=42)
kfold_score("XGB",model)



  

  