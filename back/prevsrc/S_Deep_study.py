import numpy as np

# 데이터
list_x = np.arange(1,33)
signs = np.array([1 if i%2==1 else -1 for i in range(len(list_x))])  # 홀수:+1 짝수:-1
list_y = list_x * 2 + signs

# 입력을 2개 피처로 구성: (x_norm, parity)
X_raw = list_x.reshape(-1,1).astype(float)
X = np.hstack([X_raw / 32.0, signs.reshape(-1,1)])   # x를 0~1로 정규화, parity는 그대로
y = list_y.reshape(-1,1).astype(float)

# 하이퍼파라미터
LR = 0.01
EPOCHS = 20000
H = 6   # 은닉 뉴런 수

# 활성화: tanh (은닉), 출력은 선형(회귀)
def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - np.tanh(x)**2

np.random.seed(42)
W1 = np.random.randn(2, H) * 0.1   # 입력 2 -> 은닉 H
b1 = np.zeros((1, H))
W2 = np.random.randn(H, 1) * 0.1   # 은닉 H -> 출력 1
b2 = np.zeros((1, 1))

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = tanh(Z1)
    Z2 = A1 @ W2 + b2
    return Z1, A1, Z2

def backward(X, y, Z1, A1, Z2, W1, b1, W2, b2, lr):
    m = X.shape[0]
    dZ2 = (Z2 - y) / m               # 출력층 선형
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * dtanh(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

# 학습
for epoch in range(EPOCHS):
    Z1, A1, Z2 = forward(X, W1, b1, W2, b2)
    loss = np.mean((Z2 - y)**2)
    W1, b1, W2, b2 = backward(X, y, Z1, A1, Z2, W1, b1, W2, b2, LR)
    if epoch % 5000 == 0:
        print(f"Epoch {epoch:05d} | Loss: {loss:.6f}")

print("\n학습 완료")
print("W1 (2->H):\n", W1)
print("b1:\n", b1)
print("W2 (H->1):\n", W2)
print("b2:\n", b2)

# 테스트 (testX = [6,7,8,9,10])
testX_raw = np.array([6,7,8,9,10]).reshape(-1,1).astype(float)
test_parity = np.array([1 if x%2==1 else -1 for x in testX_raw.flatten()]).reshape(-1,1)
testX = np.hstack([testX_raw/32.0, test_parity])
_, _, pred = forward(testX, W1, b1, W2, b2)
for i, x in enumerate(testX_raw.flatten()):
    print(f"입력 {int(x)} -> 예측 {pred[i,0]:.4f}  (정답 {int(x*2 + (1 if int(x)%2==1 else -1))})")
