import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plto
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# 데이터 로드 및 전처리
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()


# plt.imshow(x_train[0], cmap='gray')
# plt.show()

print(f"x_train shape: {x_train.shape},y_train shape: {y_train.shape}")

# ─────────────────────────────────────────
# 데이터 준비 
# ─────────────────────────────────────────
def preprocess_data_org(x_train, x_test):
    # 데이터 정규화 및 차원 변환
    x_train = x_train.reshape((60000, 28*28)).astype('float32') / 255.0
    x_test = x_test.reshape((10000, 28*28)).astype('float32') / 255.0
    return x_train, x_test

def preprocess_data(x_train, x_test):
    # CNN을 위해 28x28x1 형태로 차원만 변환합니다.
    # PyTorch는 (이미지 개수, 채널 수, 높이, 너비) 순서를 사용합니다.
    x_train = x_train.reshape((60000, 1, 28, 28)).astype('float32') / 255.0
    x_test = x_test.reshape((10000, 1, 28, 28)).astype('float32') / 255.0
    return x_train, x_test

x_train, x_test = preprocess_data(x_train, x_test)


# ─────────────────────────────────────────
# 모델 만들기.
# ─────────────────────────────────────────

def make_model():
    # Keras의 Sequential 모델과 유사한 PyTorch의 nn.Sequential
    # 입력 데이터: (N, 1, 28, 28)
    return nn.Sequential(
        # ─────────────────────────────────────────
        # 1. 첫 번째 Conv + ReLU + Pooling
        # ─────────────────────────────────────────
        # in_channels=1 (흑백), out_channels=32, kernel_size=3, padding=1 (크기 유지)
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # -> (N, 32, 28, 28)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # -> (N, 32, 14, 14)
        
        # ─────────────────────────────────────────
        # 2. 두 번째 Conv + ReLU + Pooling
        # ─────────────────────────────────────────
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # -> (N, 64, 14, 14)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), # -> (N, 64, 7, 7)
        
        # ─────────────────────────────────────────
        # 3. Flatten + Linear(Dense) + Softmax
        # ─────────────────────────────────────────
        nn.Flatten(),
        nn.Linear(in_features=64 * 7 * 7, out_features=10), # 64*7*7 = 3136
      #  nn.Softmax(dim=1) # 각 샘플에 대해 클래스 확률 계산
    )



#─────────────────────────────────────────
# 방법 2: nn.Module을 상속받는 클래스로 모델 정의
# 더 복잡하고 유연한 모델 구조를 만들 때 사용됩니다.

class DeepImageModel(nn.Module):
    def __init__(self):
        super(DeepImageModel,self).__init__()
        # ─────────────────────────────────────────
        # 1. 첫 번째 Conv + ReLU + Pooling
        # ─────────────────────────────────────────
        self.conv2d1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> (N,32,28,28)
        self.Relu1 = nn.ReLU()
        self.MaxPool2d1 = nn.MaxPool2d(kernel_size=2, stride=2)     # -> (N,32,14,14)

        # ─────────────────────────────────────────
        # 2. 두 번째 Conv + ReLU + Pooling
        # ─────────────────────────────────────────
        self.conv2d2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (N,64,14,14)
        self.Relu2 = nn.ReLU()
        self.MaxPool2d2 = nn.MaxPool2d(kernel_size=2, stride=2)     # -> (N,64,7,7)
        # ─────────────────────────────────────────
        # 3. Flatten + Linear(Dense) + Softmax
        # ─────────────────────────────────────────
        self.Flatten3 = nn.Flatten()
        self.Linear3 = nn.Linear(in_features=64 * 7 * 7, out_features=10)
#        self.Softmax3 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv2d1(x)
        out = self.Relu1(out)
        out = self.MaxPool2d1(out)

        out = self.conv2d2(out)
        out = self.Relu2(out)
        out = self.MaxPool2d2(out)

        out = self.Flatten3(out)
        out = self.Linear3(out)
     #   out = self.Softmax3(out)
        return out
    
#─────────────────────────────────────────
# 하이퍼 파라메터 
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# 하이퍼 파라메터 설정
# ─────────────────────────────────────────
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64

# PyTorch 텐서로 변환
x_train_tensor = torch.from_numpy(x_train)
# CrossEntropyLoss는 Long 타입의 클래스 인덱스를 기대합니다.
y_train_tensor = torch.from_numpy(y_train).long() 

x_test_tensor = torch.from_numpy(x_test)
y_test_tensor = torch.from_numpy(y_test).long()

# TensorDataset과 DataLoader를 사용하여 미니배치 생성
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ─────────────────────────────────────────
# 학습 루프  
# ─────────────────────────────────────────
def train_model(model, data_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 함수 인자 lr 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)            # logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{epochs} loss:{running_loss/len(data_loader):.4f} acc:{100*correct/total:.2f}%")

# 학습/평가 예시: 학습한 modelByClass로 평가
modelByClass = DeepImageModel()
#modelByClass = make_model()

train_model(modelByClass, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)

print("\n--- 모델 평가 (학습된 modelByClass) ---")
modelByClass.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelByClass.to(device)
x_test_tensor = x_test_tensor.to(device).float()
y_test_tensor = y_test_tensor.to(device).long()
with torch.no_grad():
    outputs = modelByClass(x_test_tensor)
    _, predicted = torch.max(outputs, 1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    print(f'테스트 정확도: {100 * correct / total:.2f} %')
