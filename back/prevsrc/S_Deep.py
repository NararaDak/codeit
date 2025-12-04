'''
+----------------+
|    입력층 (0층) |
+----------------+
      x1 = 1
      x2 = 0
         |
         V
+--------------------------------------------------+
|          1층 (은닉층) - W1, b1 학습                |
+--------------------------------------------------+
      / \
     /   \
    /     \
  W1[:,0] = (-1.5,-1.5), b1[0,0]=2.5    W1[:,1] = (-2.0,-2.0), b1[0,1]=1.0
    |                                   |
    V                                   V
[Node 1] -> z=1.0, y1=0.7310     [Node 2] -> z=-1.0, y2=0.2689
    |                                   |
    +------------------+------------------+
                       |
                       V
+--------------------------------------------------+
|           2층 (H출력층) - W2, b2 학습              |
+--------------------------------------------------+
   입력: y1=0.7310, y2=0.2689
   W2=(w3,1=-3.5, w4,1=3.5), b2[0,0]=-1.5
   z = (-3.5 * y1) + (3.5 * y2) - 1.5 ≈ 0.0
   y_XOR = Sigmoid(0.0)
                       |
                       V
                    0.5000
                    
+--------------------------------------------------+
|        최종 출력: 0.5000 (분류: 0 또는 1)        |
+--------------------------------------------------+
'''
import numpy as np
# ─────────────────────────────────────────
# 1. 하이퍼파라미터 설정
# ─────────────────────────────────────────
LEARNING_RATE = 0.1
EPOCHS = 20000

# XOR 입력과 정답
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])

H = 8  # 은닉 뉴런 수
# ─────────────────────────────────────────
# 2. 활성화 함수 정의 (ReLU + Sigmoid)
# ─────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def derivative_relu(x):
    return (x > 0).astype(float)

# ─────────────────────────────────────────
# 3. 가중치 초기화
# ─────────────────────────────────────────
def initialize_parameters():
    np.random.seed(42)
    W1 = np.random.randn(2, H) * 0.5
    b1 = np.zeros((1, H))
    W2 = np.random.randn(H, 1) * 0.5
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# ─────────────────────────────────────────
# 4. 순전파 / 역전파
# ─────────────────────────────────────────
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def back_propagation(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2):
    m = X.shape[0]
    
    # 출력층 오차
    E2 = A2 - Y
    dZ2 = E2 * derivative_sigmoid(Z2)
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # 은닉층 오차
    E1 = dZ2 @ W2.T
    dZ1 = E1 * derivative_sigmoid(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # 가중치 갱신
    W1 -= LEARNING_RATE * dW1
    b1 -= LEARNING_RATE * db1
    W2 -= LEARNING_RATE * dW2
    b2 -= LEARNING_RATE * db2

    return W1, b1, W2, b2, E2

# ─────────────────────────────────────────
# 5. 학습 루프
# ─────────────────────────────────────────
W1, b1, W2, b2 = initialize_parameters()

print("=" * 60)
print(f"XOR 게이트 학습 시작 (LR={LEARNING_RATE}, Epochs={EPOCHS})")
print("=" * 60)

for epoch in range(EPOCHS):
    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
    W1, b1, W2, b2, E2 = back_propagation(X, y, Z1, A1, Z2, A2, W1, b1, W2, b2)
    loss = np.mean(E2 ** 2)
    
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:05d} | Loss: {loss:.6f}")

# ─────────────────────────────────────────
# 6. 결과 출력
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 학습 결과")
print("=" * 60)

_, _, _, final_value = forward_propagation(X, W1, b1, W2, b2)

for i in range(len(X)):
    x1, x2 = X[i]
    prob = final_value[i][0]
    classification = 1 if prob > 0.5 else 0
    print(f"XOR({x1}, {x2}) -> 확률: {prob:.4f}, 분류: {classification} (정답: {y[i][0]})")



