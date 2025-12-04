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
# 모델 정의.
# ─────────────────────────────────────────
# 방법 1: nn.Sequential을 사용하여 함수로 모델 정의
# 간단한 순차적 모델을 만들 때 편리합니다.
def makeModel_fn():
    model = nn.Sequential(
        nn.Linear(2, HIDDEN_NEURONS),  # 입력층: 입력 피처 2개, 은닉 뉴런 HIDDEN_NEURONS개
        nn.ReLU(),                    # 활성화 함수: ReLU
        nn.Linear(HIDDEN_NEURONS, 1), # 출력층: 은닉 뉴런 HIDDEN_NEURONS개, 출력 피처 1개
        nn.Sigmoid()                  # 활성화 함수: Sigmoid (0과 1 사이의 확률값 출력)
    )
    return model

# ─────────────────────────────────────────
# 방법 2: nn.Module을 상속받는 클래스로 모델 정의
# 더 복잡하고 유연한 모델 구조를 만들 때 사용됩니다.
class DeepXORModel(nn.Module):
    # 모델의 레이어를 초기화하는 생성자
    def __init__(self, input_dim, hidden_dim, output_dim):
        # nn.Module의 생성자를 먼저 호출해야 합니다.
        super(DeepXORModel, self).__init__()
        # 첫 번째 선형 레이어 (입력층 -> 은닉층)
        self.hidden = nn.Linear(input_dim, hidden_dim) 
        # 두 번째 선형 레이어 (은닉층 -> 출력층)
        self.output = nn.Linear(hidden_dim, output_dim) 
        # 활성화 함수들을 정의합니다.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 

    # 모델의 순전파 로직을 정의하는 메서드
    def forward(self, x):
        # 입력 x를 hidden 레이어에 통과시킵니다.
        x = self.hidden(x)
        # ReLU 활성화 함수를 적용합니다.
        x = self.relu(x)
        # 결과를 output 레이어에 통과시킵니다.
        x = self.output(x)
        # 최종적으로 Sigmoid 활성화 함수를 적용하여 결과를 반환합니다.
        x = self.sigmoid(x)
        return x
model = DeepXORModel(input_dim=2, hidden_dim=HIDDEN_NEURONS, output_dim=1)
# ─────────────────────────────────────────
# 4. 학습 루프  
# ─────────────────────────────────────────
# 손실 함수 및 옵티마이저 정의(Binary Cross Entropy)    
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    