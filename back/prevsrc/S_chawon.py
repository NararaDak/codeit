from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
#────────────────────────────────────────────────────────────────────
# ▣ 원본 데이터
iris = load_iris()
X,y = iris.data, iris.target

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
#────────────────────────────────────────────────────────────────────
# ▣ 차원 축소.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(X_pca.shape)  # (150, 2)
print(X_pca[:5])  # 축소된 데이터의 일부 출력
# ▣ 데이터 train/test 분할
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)
# ▣ 로지스틱 회귀 모델 훈련
model_pca = LogisticRegression()    
model_pca.fit(X_train_pca, y_train_pca)
# 예측
y_pred_pca = model_pca.predict(X_test_pca)
# 정확도 평가
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
#────────────────────────────────────────────────────────────────────
# ▣ 차원 축소.
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
print(X_lda.shape)  # (150, 2)
print(X_lda[:5])  # 축소된 데이터의 일부 출력
# ▣ 데이터 train/test 분할
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y, test_size=0.2, random_state=42)
# ▣ 로지스틱 회귀 모델 훈련
model_lda = LogisticRegression()    
model_lda.fit(X_train_lda, y_train_lda)
# 예측  
y_pred_lda = model_lda.predict(X_test_lda)
# 정확도 평가
accuracy_lda = accuracy_score(y_test_lda, y_pred_lda)
#────────────────────────────────────────────────────────────────────
print(f'Accuracy(PCA): {accuracy_pca:.2f}')
print(f'Accuracy (LDA): {accuracy_lda:.2f}')
#────────────────────────────────────────────────────────────────────



