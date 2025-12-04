import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ================================
# 그래프 설정
# ================================
# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.style.use('fivethirtyeight')  # 그래프 스타일 설정
plt.rcParams["figure.figsize"] = 10, 6 # 기본 그래프 크기 설정

def cosine_similarity_manual(A, B):
    # 내적 계산 (dot product)
    dot_product = np.sum(A * B)
    
    # A의 크기(norm) 계산
    norm_A = np.sqrt(np.sum(A**2))
    
    # B의 크기(norm) 계산
    norm_B = np.sqrt(np.sum(B**2))
    
    # 코사인 유사도 계산
    similarity = dot_product / (norm_A * norm_B)
    
    return similarity

# 두 벡터 정의
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# 코사인 유사도 계산
similarity = cosine_similarity_manual(A, B)
print(similarity)  # 결과 출력

# 두 벡터 A와 B를 NumPy 배열로 정의
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# 유클리드 거리를 계산하는 함수
def euclidean_distance(vec_a, vec_b):
    """
    NumPy를 사용하여 두 벡터 사이의 유클리드 거리를 계산합니다.
    """
    # 1. 두 벡터의 차이를 구합니다 (B - A)
    difference = vec_b - vec_a

    # 2. 각 차이값을 제곱합니다
    squared_difference = difference**2

    # 3. 제곱된 차이값들을 모두 더합니다
    sum_squared = np.sum(squared_difference)

    # 4. 최종적으로 합산된 값의 제곱근을 구합니다 (L2 Norm)
    distance = np.sqrt(sum_squared)

    # 또는 NumPy의 내장 함수를 사용한 더 간결한 방법:
    # distance = np.linalg.norm(vec_a - vec_b)

    return distance

# 거리 계산 및 결과 출력
distance = euclidean_distance(A, B)
print(f"벡터 A: {A}")
print(f"벡터 B: {B}")
print(f"유클리드 거리: {distance}")


# 동전 던지기 시뮬레이션
trials = np.random.choice(["앞", "뒤"], size=10000, p=[0.5, 0.5])
# print(trials)

# 방법 1: Counter 사용
counts = Counter(trials)
print("앞:", counts["앞"], "개")
print("뒤:", counts["뒤"], "개")

# 방법 2: numpy를 이용한 방법
print("앞 개수:", np.count_nonzero(trials == "앞"))
print("뒤 개수:", np.count_nonzero(trials == "뒤"))


# 데이터
x = [1, 2, 3, 4, 5, 6]
label = [3, 6, 7, 12, 15, 18]

# 초기값
W = 0                # 임의의 초기 가중치
lr = 0.01              # 학습률
epochs = 100           # 반복 횟수

# 학습 기록 저장
W_list = []
loss_list = []

# 손실 함수
def compute_loss(W, x, label):
    total = 0
    n = len(x)
    for i in range(n):
        total += (label[i] - W * x[i]) ** 2
    return total / n

# 기울기 계산 함수
def compute_gradient(W, x, label):
    total = 0
    n = len(x)
    for i in range(n):
        total += -2 * x[i] * (label[i] - W * x[i])
    return total / n

# 경사 하강법 반복
for epoch in range(epochs):
    grad = compute_gradient(W, x, label)
    new_W = W - lr * grad

    # 기록
    W_list.append(W)
    loss_list.append(compute_loss(W, x, label))

    # 출력 (감소/증가 여부 표시)
    direction = "⬇ 감소" if grad > 0 else "⬆ 증가"
    print(f"{epoch+1:3d} | W = {W:.4f} | 손실 = {loss_list[-1]:.4f} | dL/dW = {grad:.4f} | {direction}")

    # 업데이트
    W = new_W

# 최종 W 출력
print("\n최적의 W:", round(W, 4))

# 그래프 시각화
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(W_list, loss_list, marker='o')
plt.title("W 변화에 따른 손실(Loss)")
plt.xlabel("W")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(W_list, marker='o', color='purple')
plt.title("W 업데이트 과정")
plt.xlabel("Epoch")
plt.ylabel("W 값")

plt.tight_layout()
plt.show()

TP = 5
FN = 5
FP = 5
TN = 85

acur = (TP + TN) / 100
Pre = TP / (TP + FP)
Recl = TP / (TP + FN)
F1 = 2 * (Pre * Recl) / (Pre + Recl)

print("Accuracy:", acur)
print("Precision:", Pre)
print("Recall:", Recl)
print("F1-score:", F1)