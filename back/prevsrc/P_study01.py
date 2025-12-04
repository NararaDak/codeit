import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression



# 초기값
W = 0.0          # 초기 가중치 (Weight)
b = 0.0          # 초기 편향 (Bias)
lr = 0.01        # 학습률
epochs = 1000    # 반복 횟수를 넉넉하게 늘립니다.

# 학습 기록 저장
W_list = []
b_list = []
loss_list = []

# 데이터 (공부 시간 -> 점수)
# 여전히 y = 2x 관계이지만, b가 0에 수렴하는지 확인합니다.
X = np.array([1, 2, 3, 4, 5], dtype=float).reshape(-1, 1) # (5, 1) 형태의 2차원 배열로 변환
label = np.array([2, 4, 6, 8, 10], dtype=float).reshape(-1, 1) # (5, 1) 형태의 2차원 배열로 변환


# 손실 함수 (H(x) = Wx + b)
def compute_loss(W, b, x, label):
    n = len(x)
    hypothesis = W * x + b
    # 평균 제곱 오차 (MSE)
    return np.sum((label - hypothesis) ** 2) / n

# 기울기 계산 함수
def compute_gradient(W, b, x, label):
    n = len(x)
    hypothesis = W * x + b
    error = label - hypothesis
    
    # W에 대한 기울기 (dL/dW)
    # dL/dW = 1/n * Σ(-2x * (y - H(x)))
    grad_W = np.sum(-2 * x * error) / n
    
    # b에 대한 기울기 (dL/db)
    # dL/db = 1/n * Σ(-2 * (y - H(x)))
    grad_b = np.sum(-2 * error) / n
    
    return grad_W, grad_b

# 경사 하강법 반복
print("--- 경사 하강법 학습 과정 (W, b 포함) ---")
for epoch in range(epochs):
    # 1. 기울기 계산
    grad_W, grad_b = compute_gradient(W, b, X, label)

    # 2. W와 b 갱신 (경사 하강)
    new_W = W - lr * grad_W
    new_b = b - lr * grad_b

    # 3. 기록 및 출력
    current_loss = compute_loss(W, b, X, label)
    W_list.append(W)
    b_list.append(b)
    loss_list.append(current_loss)
    
    if (epoch + 1) % 100 == 0: # 100 에포크마다 출력
        print(f"Epoch {epoch+1:4d} | W: {W:.4f} | b: {b:.4f} | Loss: {current_loss:.6f}")

    # 4. 다음 반복을 위해 파라미터 업데이트
    W = new_W
    b = new_b

print("\n-------------------------------------")
print(f"✅ 학습 완료 (Epochs: {epochs}, Learning Rate: {lr})")
print(f"최종 W (가중치): {W:.8f}")
print(f"최종 b (편향): {b:.8f}")
print(f"최종 Loss (손실): {compute_loss(W, b, X, label):.10f}")

x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10])

X = x.reshape(-1,1)
print("\n--- scikit-learn을 이용한 선형 회귀 ---")
# scikit-learn 모델은 2차원 입력을 기대하므로 X는 이미 올바른 형태입니다.
# y는 1차원 형태를 선호하므로 label.ravel() 또는 y를 새로 정의하여 사용합니다.
y_train = label.ravel()
model = LinearRegression()
model.fit(X,y)
model.fit(X, y_train)

print("Predict ratio",model.predict_proba([3],[4],[5]))
print("Predict",model.predict([3],[4],[5]))
print(f"scikit-learn이 찾은 W: {model.coef_[0]:.8f}")
print(f"scikit-learn이 찾은 b: {model.intercept_:.8f}")



# 예측할 값들을 2차원 배열로 전달해야 합니다.
predict_values = np.array([[3], [4], [5]])
predictions = model.predict(predict_values)
print(f"\n입력값 [3, 4, 5]에 대한 예측: {predictions}")
