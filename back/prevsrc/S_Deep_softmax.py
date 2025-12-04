import numpy as np

# ─────────────────────────────────────────
# 0. Softmax 및 교차 엔트로피 손실 함수 정의
# ─────────────────────────────────────────
def softmax(x):
    # 오버플로우 방지를 위해 최댓값을 뺀 후 exp 계산
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(A2, Y):
    m = Y.shape[0]
    # Y는 원-핫 인코딩된 정답 레이블
    log_probs = -np.log(A2[range(m), np.argmax(Y, axis=1)] + 1e-10) # 로그 0 방지
    loss = np.sum(log_probs) / m
    return loss

# ─────────────────────────────────────────
# 1. 하이퍼파라미터 설정
# ─────────────────────────────────────────
LEARNING_RATE = 0.1
EPOCHS = 20000

# XOR 입력
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# 정답 (One-hot 인코딩으로 변경: [0] -> [1, 0], [1] -> [0, 1])
# 클래스 0: [1, 0], 클래스 1: [0, 1]
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]]) 

H = 8  # 은닉 뉴런 수
OUTPUT_DIM = 2 # 출력 뉴런 수 (클래스 수)

# ─────────────────────────────────────────
# 2. 활성화 함수 정의 (ReLU + Softmax)
# (기존 sigmoid 및 derivative_sigmoid는 사용하지 않음)
# ─────────────────────────────────────────
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
    # W2의 형태를 (H, 1) -> (H, OUTPUT_DIM)로 변경
    W2 = np.random.randn(H, OUTPUT_DIM) * 0.5 
    b2 = np.zeros((1, OUTPUT_DIM))
    return W1, b1, W2, b2

# ─────────────────────────────────────────
# 4. 순전파 / 역전파 (Softmax + Cross-Entropy)
# ─────────────────────────────────────────
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2) # 출력층에 Softmax 적용
    return Z1, A1, Z2, A2

def back_propagation(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2):
    m = X.shape[0]
    
    # 출력층 오차 (Softmax + Cross-Entropy 미분)
    # E2는 A2 - Y와 동일 (매우 간단해짐)
    E2 = A2 - Y 
    
    # 2단계: 출력층 가중치 업데이트
    dW2 = A1.T @ E2 / m
    db2 = np.sum(E2, axis=0, keepdims=True) / m

    # 3단계: 은닉층 오차 계산 (Softmax 미분이 아닌 E2를 사용)
    E1 = E2 @ W2.T
    dZ1 = E1 * derivative_relu(Z1) # ReLU 미분 사용
    
    # 4단계: 은닉층 가중치 업데이트
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
print(f"XOR 게이트 학습 시작 (Softmax + Cross-Entropy) (LR={LEARNING_RATE}, Epochs={EPOCHS})")
print("=" * 60)

for epoch in range(EPOCHS):
    Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
    W1, b1, W2, b2, E2 = back_propagation(X, y, Z1, A1, Z2, A2, W1, b1, W2, b2)
    # 손실 계산에 cross_entropy_loss 사용
    loss = cross_entropy_loss(A2, y) 
    
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
    # Softmax 출력은 두 클래스에 대한 확률 벡터 [P(Class 0), P(Class 1)]
    prob_vec = final_value[i] 
    # 가장 높은 확률을 가진 인덱스를 분류 결과로 취함
    classification = np.argmax(prob_vec) 
    
    # 정답 레이블을 원-핫 인코딩에서 스칼라 값으로 다시 변환
    target_class = np.argmax(y[i])
    
    print(f"XOR({x1}, {x2}) -> 확률 벡터: [{prob_vec[0]:.4f}, {prob_vec[1]:.4f}], 분류: {classification} (정답: {target_class})")