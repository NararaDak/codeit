import pandas as pd
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
warnings.filterwarnings('ignore')

# data import
train = pd.read_csv('d:/temp/titanic_train.csv')
test = pd.read_csv('d:/temp/titanic_test.csv')
#gender_submission = pd.read_csv('data/gender_submission.csv')
train['Sex_clean'] = train['Sex'].astype('category').cat.codes
test['Sex_clean'] = test['Sex'].astype('category').cat.codes
train['Embarked'].isnull().sum()
# 2

test['Embarked'].isnull().sum()
# 0

train['Embarked'].value_counts()
# output
# S    644
# C    168
# Q     77


def age_by_custom_ranges(df: pd.DataFrame, scol,tcol ):
    df.loc[ df[scol] <= 10, tcol] = 0
    df.loc[(df[scol] > 10) & (df[scol] <= 16), tcol] = 1
    df.loc[(df[scol] > 16) & (df[scol] <= 20), tcol] = 2
    df.loc[(df[scol] > 20) & (df[scol] <= 26), tcol] = 3
    df.loc[(df[scol] > 26) & (df[scol] <= 30), tcol] = 4
    df.loc[(df[scol] > 30) & (df[scol] <= 36), tcol] = 5
    df.loc[(df[scol] > 36) & (df[scol] <= 40), tcol] = 6
    df.loc[(df[scol] > 40) & (df[scol] <= 46), tcol] = 7
    df.loc[(df[scol] > 46) & (df[scol] <= 50), tcol] = 8
    df.loc[(df[scol] > 50) & (df[scol] <= 60), tcol] = 9
    df.loc[ df[scol] > 60, tcol] = 10


def preprocess_data(df):
    df['Embarked'].fillna('S', inplace=True)

    df['Embarked_clean'] = df['Embarked'].astype('category').cat.codes
    df['Family'] = 1 + df['SibSp'] + df['Parch']
    df['Solo'] = (df['Family'] == 1)
    df['Fare'] = df['Fare'].apply(lambda x : 10 * np.log(x) if x>0 else 0)

    df['FareBin'] = pd.qcut(df['Fare'], 5)
    
    df['FareBin'].value_counts()
    df['Fare_clean'] = df['FareBin'].astype('category').cat.codes
    
    df['Fare_clean'].value_counts()
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    df['Title'].value_counts()
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df['Title'].value_counts()
    
    df['Title_clean'] = df['Title'].astype('category').cat.codes
    df['Age'].isnull().sum()
    
    df["Age"].fillna(df.groupby("Title")["Age"].transform("median"), inplace=True)
    age_by_custom_ranges(df,"Age","Age_clean")
    df['Cabin'].str[:1].value_counts()
    mapping = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'T': 7
    }
    df['Cabin_clean'] = df['Cabin'].str[:1]
    df['Cabin_clean'] = df['Cabin_clean'].map(mapping)
    df['Cabin_clean'] = df.groupby('Pclass')['Cabin_clean'].transform('median')
    df["Pclass_Agemean"] =  df.groupby("Pclass")["Age"].transform("mean")
    age_by_custom_ranges(df,"Pclass_Agemean","Pclass_Agemean_clean")
    df["Title_Agemean"] =  df.groupby("Title")["Age"].transform("mean")
    age_by_custom_ranges(df,"Title_Agemean","Title_Agemean_Clean")
    df["Pclass_Family"] = df.groupby("Pclass")["Family"].transform("mean")
    df["Pclass_Cabin_clean"] = df.groupby("Pclass")["Cabin_clean"].transform("mean")

    
   
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
   # 'Fare_clean',
    'Fare',
    'Pclass_Family',
]

label = [
    'Survived',
]

preprocess_data(train)
preprocess_data(test)

def save_predict(y_predict,savename):
    submission = pd.DataFrame({
                "PassengerId":test["PassengerId"],
                "Survived": y_predict
            })
    submission.to_csv(f"d:/temp/Submission_{savename}.csv",index=False)

def predict_Random():
    data = train[feature]
    target = train[label]

    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    #--------------------------------------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=0)
    print("=======================================================================")
    print("RandomForestClassifier:", cross_val_score(clf, data, target, cv=k_fold, scoring='accuracy', ).mean())
    print("=======================================================================")
    #--------------------------------------------------------------------------------------
    x_train = train[feature]
    x_test = test[feature]
    y_train = train[label]
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    save_predict(y_predict,"Random01")
    #--------------------------------------------------------------------------------------
def predict_XGB(nestimators=300,learningrate=0.01,maxdepth=5):
    data = train[feature]
    target = train[label]
    #--------------------------------------------------------------------------------------
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = XGBClassifier(n_estimators=nestimators, learning_rate= learningrate, max_depth=maxdepth, eval_metric='logloss', use_label_encoder=False, random_state=42)
    #--------------------------------------------------------------------------------------
    print("=======================================================================")
    print(f"{nestimators},{learningrate},{maxdepth},XGBClassifier:{cross_val_score(clf, data, target, cv=k_fold, scoring='accuracy', ).mean()}")
    print("=======================================================================")
    #--------------------------------------------------------------------------------------
    x_train = train[feature]
    x_test = test[feature]
    y_train = train[label]
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    # save_predict(y_predict,"XGB01")
 
# predict_Random()
predict_XGB(300,0.1,5)
predict_XGB(300,0.01,5)
predict_XGB(300,0.001,5)
predict_XGB(300,0.0001,5)




