import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from filelock import FileLock
import os
import datetime
import sys
import torch
import torch.nn as nn


# ════════════════════════════════════════
# ▣ Meta/유틸리티함수.
# ════════════════════════════════════════
ver = "2025.12.01.001"
#BASE_DIR = r"D:\01.project\CodeIt\mission8\data"
#BASE_DIR = "/content/drive/MyDrive/codeit/mission8/data"
#BASE_DIR = r"d:\01.project\codeitmission8\mission8\data"
BASE_DIR = r"D:\01.project\CodeIt\data"
LOG_FILE = f"{BASE_DIR}/m9log.txt"
RESULT_CSV = f"{BASE_DIR}/m9result.csv"
BASE_DIR = f"{BASE_DIR}/mission9"

## 구분선 출력 함수
def Lines(text="", count=100):
    print("═" * count)
    if text != "":
        print(f"{text}")
        print("═" * count)
## 현재 시간 문자열 반환 함수
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
## 디렉토리 생성 함수
def makedirs(d):
    os.makedirs(d, exist_ok=True)
## 운영 로그 함수
def OpLog(log, bLines=True):
    if bLines:
        Lines(f"[{now_str()}] {log}")
    try:
        caller_name = sys._getframe(1).f_code.co_name
    except Exception:
        caller_name = "UnknownFunction"
        
    log_filename = LOG_FILE
    log_lock_filename = log_filename + ".lock"
    log_content = f"[{now_str()}] {caller_name}: {log}\n"
    try:
        lock = FileLock(log_lock_filename, timeout=10)
        with lock:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(log_content)
    except Exception as e:
        print(f"로그 파일 쓰기 오류 발생: {e}")

OpLog(f"Program started.{ver}", bLines=True)


## 그래픽 출력 함수
def ShowPlt(plt):
    # plt.tight_layout()
    # plt.show(block = False)
    # plt.pause(3)
    plt.close()
   
## 메타 클래스 - 전역 설정 및 데이터 정보 관리
class MyMeta():
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def device(self):
        return self._device

MY_META = MyMeta()



# ════════════════════════════════════════
# ▣ LOADER.
# ═══════════════════════════════════════
def GetLoader(transform_type):

    # Transform 타입별 데이터 증강 전략:
    # - A: 기본 (정규화만)
    # - B: 중간 증강 (Flip, Rotation, Affine)
    # - C: 강력 증강 (B + Random Erasing, Elastic Transform)

    if transform_type == "A":
        # 타입 A: 기본 변환 (정규화만)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1, 1] 범위로 정규화
        ])
    
    elif transform_type == "B":
        # 타입 B: 중간 수준 데이터 증강
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
            transforms.RandomRotation(degrees=15),    # ±15도 회전
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 10% 이동
                scale=(0.9, 1.1),      # 90~110% 크기 조정
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    elif transform_type == "C":
        # 타입 C: 강력한 데이터 증강
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),    # ±20도 회전 (B보다 큼)
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),  # 15% 이동 (B보다 큼)
                scale=(0.8, 1.2),        # 80~120% 크기 조정 (B보다 넓음)
                shear=10,                # ±10도 전단 변환 추가
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # 원근 변환
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # 30% 확률로 일부 영역 제거
        ])
    
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}. Choose 'A', 'B', or 'C'.")

    train_dataset = torchvision.datasets.FashionMNIST(root=BASE_DIR, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root=BASE_DIR, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# ════════════════════════════════════════
# ▣ Class.
# ═══════════════════════════════════════
# 수정 이유:
# 1. Conv1d → Conv2d: 이미지 데이터는 2D(높이×너비)이므로 Conv2d 사용 필요
# 2. BatchNorm1d → BatchNorm2d: Conv2d와 함께 사용하려면 2D BatchNorm 필요
# 3. LeakyReLU(-1.2) → LeakyReLU(0.2): negative_slope는 양수여야 함 (일반적으로 0.01~0.3)
# 4. torch.cat(..., -2) → torch.cat(..., -1): 마지막 차원(feature dim)에서 결합
# 5. img.size(-1) → img.size(0): 배치 크기는 첫 번째 차원(0)
# 6. torch.cat(..., 0) → torch.cat(..., 1): 채널 차원(1)에서 이미지와 레이블 결합

class BaseGan(nn.Module):
    # GAN 모델의 기본 클래스 - 공통 변수 및 학습 로직
    def __init__(self, latent_dim=100):
        super(BaseGan, self).__init__()
        self._image_size = 28      # Fashion MNIST 이미지 크기
        self._num_classes = 10     # 10개 클래스
        self._latent_dim = latent_dim  # 잠재 공간 차원 (파라미터로 받음)
        
        self._generator = None
        self._discriminator = None
        self._criterion = nn.BCELoss()
        self._lr = None 
        self._optimizer_G = None
        self._optimizer_D = None
        self._betas = None
        self._epochs = None
        self._trans_type = None
        self._idx_to_class = None
        self._train_loader = None
        self._test_loader = None

    def fit(self, trans_type, trainLoader, testLoader, epochs=30, lr=0.0002, betas=(0.5, 0.999)):
        """GAN 모델 학습"""
        self._trans_type = trans_type
        self._idx_to_class = {i: class_name for i, class_name in enumerate(trainLoader.dataset.classes)}
        self._train_loader = trainLoader
        self._test_loader = testLoader
        self._epochs = epochs
        self._lr = lr
        self._betas = betas
        self._optimizer_G = optim.Adam(self._generator.parameters(), lr=lr, betas=betas)
        self._optimizer_D = optim.Adam(self._discriminator.parameters(), lr=lr, betas=betas)

        for epoch in range(epochs):
            for i, (imgs, labels) in enumerate(self._train_loader):
                batch_size_current = imgs.size(0)
                imgs = imgs.to(MY_META.device())
                labels = labels.to(MY_META.device())

                valid = torch.ones(batch_size_current, 1, device=MY_META.device())
                fake  = torch.zeros(batch_size_current, 1, device=MY_META.device())

                # 판별자 학습
                self._optimizer_D.zero_grad()
                real_loss = self._criterion(self._discriminator(imgs, labels), valid)

                noise = torch.randn(batch_size_current, self._latent_dim, device=MY_META.device())
                gen_labels = labels
                gen_imgs = self._generator(noise, gen_labels)
                fake_loss = self._criterion(self._discriminator(gen_imgs.detach(), labels), fake)
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self._optimizer_D.step()

                # 생성자 학습
                self._optimizer_G.zero_grad()
                g_loss = self._criterion(self._discriminator(gen_imgs, labels), valid)
                g_loss.backward()
                self._optimizer_G.step()
                msg = f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(self._train_loader)}] "
                msg += f"D_loss: {d_loss.item():.4f}  G_loss : {g_loss.item():.4f}"
                OpLog(msg, bLines=False)
                print(msg, end="\r")

            # 10 에포크마다 생성 이미지 시각화 (각 클래스별로 3개씩 출력)
            if (epoch + 1) % 10 == 0:
                self._visualize_results(epoch, epochs)

    def _visualize_results(self, epoch, epochs):
        """생성 이미지 시각화"""
        self._generator.eval()
        n_row = 3               # 각 클래스당 3개씩
        n_col = self._num_classes     # 총 10개 클래스
        total_samples = n_row * n_col
        noise = torch.randn(total_samples, self._latent_dim, device=MY_META.device())
        labels_sample = torch.arange(0, self._num_classes, device=MY_META.device()).repeat(n_row)
        gen_imgs = self._generator(noise, labels_sample).detach().cpu()

        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
        for i in range(n_row):
            for j in range(n_col):
                idx = i * n_col + j
                axs[i, j].imshow(gen_imgs[idx, 0, :, :], cmap='gray')
                axs[i, j].axis('off')
                if i == 0:
                    axs[i, j].set_title(self._idx_to_class[j], fontsize=10)
        plt.tight_layout()
        plt.show()
        self._generator.train()


class CustomGanModel(BaseGan):
    # 생성자 (Generator)
    class Generator(nn.Module):
        def __init__(self, latent_dim, num_classes, image_size):
            super(CustomGanModel.Generator, self).__init__()
            self.latent_dim = latent_dim
            self.num_classes = num_classes
            self.image_size = image_size
            
            # 레이블 임베딩
            self.label_emb = nn.Embedding(num_classes, num_classes)

            self.init_size = image_size // 4  # 7

            self.l1 = nn.Sequential(
                nn.Linear(latent_dim + num_classes, 128 * self.init_size * self.init_size),
                nn.ReLU(inplace=True)
            )

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),  # Conv2d 출력용 2D BatchNorm
                nn.Upsample(scale_factor=2),  # 7 → 14
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # 2D 이미지 처리
                nn.BatchNorm2d(64, 0.8),  # Conv2d 출력용 2D BatchNorm
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2),  # 14 → 28
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # 2D 이미지 처리
                nn.Tanh()  # 출력 범위 [-1, 1]
            )

        def forward(self, noise, labels):
            label_input = self.label_emb(labels)
            gen_input = torch.cat((noise, label_input), -1)  # 마지막 차원(feature)에서 결합
            out = self.l1(gen_input)
            out = out.view(out.size(0), 128, self.init_size, self.init_size)  # size(0): 배치 크기
            img = self.conv_blocks(out)
            return img

    # 판별자 (Discriminator)
    class Discriminator(nn.Module):
        def __init__(self, num_classes, image_size):
            super(CustomGanModel.Discriminator, self).__init__()
            self.num_classes = num_classes
            self.image_size = image_size
            
            # 레이블을 단일 채널 값으로 임베딩
            self.label_emb = nn.Embedding(num_classes, 1)

            self.model = nn.Sequential(
                # 이미지(1 채널)와 레이블(1 채널)을 채널 차원에서 결합하여 2채널로 입력
                nn.Conv2d(1 + 1, 64, kernel_size=3, stride=2, padding=1),  # 2D 이미지용, 28 → 14
                nn.LeakyReLU(0.2, inplace=True),  # negative_slope=0.2 (양수, 일반적 범위: 0.01~0.3)

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 2D 이미지용, 14 → 7
                nn.BatchNorm2d(128),  # Conv2d 출력용 2D BatchNorm
                nn.LeakyReLU(0.2, inplace=True),  # negative_slope=0.2

                nn.Flatten(),
                nn.Linear(128 * (image_size // 4) * (image_size // 4), 1),   # 128*7*7
                nn.Sigmoid()
            )

        def forward(self, img, labels):
            batch_size = img.size(0)  # 첫 번째 차원(0)이 배치 크기
            label = self.label_emb(labels)
            label = label.view(batch_size, 1, 1, 1)
            label = label.expand(batch_size, 1, self.image_size, self.image_size)
            # 이미지와 레이블을 채널 차원(1)에서 연결
            d_in = torch.cat((img, label), 1)  # dim=1: 채널 차원
            validity = self.model(d_in)
            return validity
        
    def __init__(self, latent_dim=100):
        super(CustomGanModel, self).__init__(latent_dim)
        # Generator와 Discriminator 생성 시 파라미터 전달
        self._generator = self.Generator(self._latent_dim, self._num_classes, self._image_size).to(MY_META.device())
        self._discriminator = self.Discriminator(self._num_classes, self._image_size).to(MY_META.device())

    
        
# 모델 초기화 및 학습 실행
# model = CustomGanModel()
# model.fit(epochs=30)

def Single_Train(trans_type='A', epochs=30, lr=0.0002, betas=(0.5, 0.999)):
    model = CustomGanModel()
    train_loader, test_loader = GetLoader(transform_type=trans_type)
    model.fit(trans_type=trans_type, trainLoader=train_loader, testLoader=test_loader, epochs=epochs, lr=lr, betas=betas)

Single_Train()

