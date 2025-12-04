import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def Lines():
    print("\n" + "─" * 90)
# ─────────────────────────────────────────
# 1. XOR 데이터 준비
# ─────────────────────────────────────────
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
# ─────────────────────────────────────────
# 2. 하이퍼 파라메터 설정
# ─────────────────────────────────────────
LEARNING_RATE = 0.1
EPOCHS = 10000
HIDDEN_NEURONS = 8
# ─────────────────────────────────────────
# 3. 모델 정의 (은닉층 1개, ReLU 활성화, 출력층 Sigmoid)
# ─────────────────────────────────────────
def makeModel():
    model = nn.Sequential(
        nn.Linear(2, HIDDEN_NEURONS),
        nn.ReLU(),
        nn.Linear(HIDDEN_NEURONS, 1),
        nn.Sigmoid()
    )
    return model


class DeepXORModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepXORModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

#model = DeepXORModel(input_dim=2, hidden_dim=HIDDEN_NEURONS, output_dim=1)
model = makeModel()

# ─────────────────────────────────────────
# 4. 학습 루프
# ─────────────────────────────────────────
# 손실 함수 및 옵티마이저 정의(Binary Cross Entropy)
def train_loop():
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad() # 이전 epoch의 기울기 초기화.
        # 순전파
        outputs = model(X_tensor)
        # 손실계산
        loss = criterion(outputs, y_tensor)
        # 역전파 및 최적화
        loss.backward() # 역전파를 통해 기울기 계산.
        optimizer.step() # 가중치 업데이트.
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:05d} | Loss: {loss.item():.6f}")
        if loss.item() <= 0.00001:
            print(f"학습 종료: Epoch {epoch:05d} | Loss: {loss.item():.6f}")
            break

def train():
    model.fit(X_tensor, y_tensor, epochs=EPOCHS, lr=LEARNING_RATE)
train_loop()
# ─────────────────────────────────────────
# 5. 결과 출력
# ─────────────────────────────────────────
print("\n학습 완료\n")
with torch.no_grad():
    final_outputs = model(X_tensor)
    predicted = (final_outputs.numpy() > 0.5).astype(int)
    print("입력 데이터:\n", X)
    print("예측 결과:\n", predicted)
    print("실제 값:\n", y)
    print("최종 출력 값:\n", final_outputs.numpy())
# ─────────────────────────────────────────
# 6. 테스트
# ─────────────────────────────────────────
model.eval()

with torch.no_grad():
    outputs = model(X_tensor)
    predicted = (outputs.numpy() > 0.5).astype(float)
    for i in range(len(X)):
        x1, x2 = X[i]
        prob = outputs[i].item()
        classification = int(predicted[i].item())
        target = int(y[i][0])
        print(f"입력: ({x1}, {x2}) | 예측 확률: {prob:.4f} | 분류: {classification} | 정답: {target}")
# ─────────────────────────────────────────
# 7. 파라메터 확인.
# ─────────────────────────────────────────
Lines()
# 2. parameters() 메서드 사용
print("--- parameters()를 통한 확인 ---")
for name, param in model.named_parameters():
    if param.requires_grad:
        Lines()
        print(f"Parameter Name: {name}")
        Lines()
        print(f"Shape: {param.data.shape}")
        print(param.data.numpy())
        Lines()
       