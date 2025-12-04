import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
import torchvision
import torch.optim as optim
from   torchvision import datasets,transforms
from  torch.utils.data import DataLoader,random_split
from  torchvision.transforms import v2

print("torch version:", torch.__version__)
print("Torchvison version",torchvision.__version__)

train_dataset = datasets.MNIST

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5,), std=(0.5,))
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"학습 데이터 갯수: {len(train_dataset)}")
print(f"테스트 데이터 갯수: {len(test_dataset)}")

def view_data(image,label):
    image = image.squeeze()
    class_name = range(10)
    plt.figure(figsize=(2,2))
    plt.imshow(image,cmap="gray")
    plt.xlabel(class_name[label])
    plt.show()

for i in range(3):
    image = train_dataset[i][0]
    label = train_dataset[i][1]
    view_data(image,label)


class BasicAutoencoder(nn.Module):
    def __init__(self):
        super(BasicAutoencoder, self).__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(), 
            # nn.Tanh()  # Normalize에 -1 ~ 1 범위로 했으므로 Tanh 사용
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 모델, 손실 함수, 옵티마이저 정의
model = BasicAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 학습
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images,_ in train_loader:
        img = images.view(images.size(0), -1)
        # 순전파
        output = model(img)
        loss = criterion(output, img)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 테스트
model.eval()
with torch.no_grad():
    for images,_ in test_loader:
        images = images.view(images.size(0), -1)
        output = model(images)
        break

    images = images.view(-1,28,28)
    output = output.view(-1,28,28)
    fig,axes = plt.subplots(nrows=2,ncols=10,sharex=True,sharey=True,figsize=(15,6))
    for i in range(10):
        axes[0][i].imshow(images[i],cmap="gray")
        axes[0][i].set_title(f"Original:{i+1}")
        axes[0][i].axis('off')
        axes[1][i].imshow(output[i],cmap="gray")
        axes[1][i].set_title(f"Reconstructed:{i+1}")
        axes[1][i].axis('off')
    plt.tight_layout()
    plt.show()



 



    
