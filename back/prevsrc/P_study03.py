import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# scikit-learn 모델의 2차원 입력 형태로 변환
X = x.reshape(-1, 1)

print("--- scikit-learn을 이용한 선형 회귀 ---")

model = LinearRegression()
model.fit(X, y)

# 예측할 새로운 데이터 정의 및 2차원 형태로 변환
# 예측 입력값: x=3, x=4, x=5
X_new = np.array([3, 4, 5]).reshape(-1, 1) 

# 수정된 예측 호출: predict_proba는 제거하고, predict에 올바른 2차원 입력 전달
predictions = model.predict(X_new) 

print("Predict:", predictions)
print(f"scikit-learn이 찾은 W: {model.coef_[0]:.8f}")
print(f"scikit-learn이 찾은 b: {model.intercept_:.8f}")


x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10]) 
w = 0
b = 0
lr = 0.01
epochs = 1000

for n in range(epochs):
    y_pred = w*x + b
    loss = y_pred - y
    dw = (2/len(x)) * np.dot(loss,X)
    db = (2/len(x)) * np.sum(loss)
    w -= lr*dw
    b -= lr*db
    if (n % 100) == 0:
        print(f"n:{n},w:{w},b:{b}")

print(f"w:{w},b:{b}")


X = np.array([1,2,3,4,5])
y = np.array([0,0,0,1,1])

w = 0.0
b = 0.0
lr = 0.01
epochs = 10000

def sigmoid_fn(val):
    return 1/(1+np.exp(-val))

for i in range(epochs):
    z = w*X + b
    y_pred = sigmoid_fn(z)
    loss = y_pred - y
    dw = np.dot(loss,X)/len(X)
    db = np.sum(loss)/len(X)
    w -= lr*dw
    b -= lr*db
    if (i % 100) == 0:
        print( f"i:{i},w:{w},b:{b}")

test = np.array([1,2,3,4,5])
probs = sigmoid_fn(w*test+b)
predictions = (probs >= 0.5).astype(int)
print("probs(확율):",probs)
print("분류",predictions)

x_org = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1])

model = LogisticRegression()
model.fit(x_org, y)

print("예측 확률:", model.predict_proba([[3], [4], [5]]))
print("분류 결과:", model.predict([[3], [4], [5]]))


