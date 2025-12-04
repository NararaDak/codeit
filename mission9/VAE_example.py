"""
VAE (Variational Autoencoder) - 고양이 이미지 노이즈 제거
- 이미지를 128차원 잠재 공간으로 압축했다가 복원
- 노이즈는 학습되지 않아 자동으로 제거됨
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import os

# 이미지 전처리: 64×64 크기로 리사이즈 후 텐서 변환
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 모든 이미지를 64×64 크기로 통일
    transforms.ToTensor()         # PIL 이미지 → PyTorch 텐서 (0~1 범위)
])

# 커스텀 데이터셋: 폴더 내 이미지 파일을 직접 로드
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # 지정된 폴더에서 jpg, png, jpeg 파일 경로 수집
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                          glob.glob(os.path.join(image_dir, "*.png")) + \
                          glob.glob(os.path.join(image_dir, "*.jpeg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # RGB 3채널로 변환
        if self.transform:
            image = self.transform(image)
        return image, 0  # VAE는 레이블 불필요 (비지도 학습)

# 고양이 이미지 데이터셋 로드
dataset = ImageDataset(image_dir=r"D:\01.project\CodeIt\data\catanddog\cats", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
RESULT_DIR = r"D:\01.project\CodeIt\data\catanddog\vae_result"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional VAE (합성곱 신경망 기반 변분 오토인코더)
class CVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim  # 잠재 공간 차원 (이미지 압축 크기)

        # ============ Encoder: 이미지 → 잠재 공간 ============
        # CNN으로 이미지의 핵심 특징 추출 (64×64 → 4×4)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 3채널 → 32채널, 크기 절반 (64→32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32채널 → 64채널 (32→16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 64채널 → 128채널 (16→8)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 128채널 → 256채널 (8→4)
            nn.ReLU(),
        )
        # 최종 출력: 256×4×4 = 4096차원

        # 확률 분포의 평균(μ) 계산: 4096차원 → 128차원
        self.enc_fc_mu = nn.Linear(256*4*4, latent_dim)
        
        # 확률 분포의 로그 분산(log σ²) 계산: 4096차원 → 128차원
        # 로그 분산을 사용하는 이유: 분산은 항상 양수여야 하므로 exp(log σ²)로 변환
        self.enc_fc_logvar = nn.Linear(256*4*4, latent_dim)

        # ============ Decoder: 잠재 공간 → 이미지 ============
        # 잠재 벡터를 CNN 입력 크기로 확장: 128차원 → 4096차원
        self.dec_fc = nn.Linear(latent_dim, 256*4*4)

        # Transposed Convolution으로 이미지 복원 (4×4 → 64×64)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 256채널 → 128채널 (4→8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 128채널 → 64채널 (8→16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64채널 → 32채널 (16→32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 32채널 → 3채널 (32→64)
            nn.Sigmoid()  # 출력값을 0~1 범위로 제한 (이미지 픽셀 범위)
        )

    def encode(self, x):
        """이미지를 잠재 공간의 확률 분포(μ, σ²)로 인코딩"""
        h = self.enc(x)                 # CNN 특징 추출: (B, 3, 64, 64) → (B, 256, 4, 4)
        h = h.view(h.size(0), -1)       # 평탄화: (B, 256, 4, 4) → (B, 4096)
        mu = self.enc_fc_mu(h)          # 평균 벡터: (B, 128)
        logvar = self.enc_fc_logvar(h)  # 로그 분산 벡터: (B, 128)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization Trick: 역전파 가능한 샘플링
        
        z ~ N(μ, σ²) 를 z = μ + ε×σ 로 변환 (ε ~ N(0,1))
        - 직접 샘플링하면 역전파 불가
        - μ와 σ를 분리하여 역전파 가능하게 만듦
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 × log σ²) = √σ²
        eps = torch.randn_like(std)     # 표준정규분포 노이즈 ε ~ N(0,1)
        return mu + eps * std           # z = μ + ε×σ

    def decode(self, z):
        """잠재 벡터를 이미지로 디코딩"""
        h = self.dec_fc(z)           # 선형 변환: (B, 128) → (B, 4096)
        h = h.view(-1, 256, 4, 4)    # 재구성: (B, 4096) → (B, 256, 4, 4)
        out = self.dec(h)            # Transposed Conv: (B, 256, 4, 4) → (B, 3, 64, 64)
        return out

    def forward(self, x):
        """전체 VAE 처리: 이미지 → 잠재 공간 → 복원 이미지"""
        mu, logvar = self.encode(x)          # 인코딩: 확률 분포 파라미터
        z = self.reparameterize(mu, logvar)  # 샘플링: 잠재 벡터
        out = self.decode(z)                 # 디코딩: 복원 이미지
        return out, mu, logvar

# 모델 및 옵티마이저 초기화
model = CVAE(latent_dim=128)  # 128차원 잠재 공간 (압축률: 4096→128, 약 32배)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon, x, mu, logvar):
    """VAE 손실 함수 = 복원 손실 + KL Divergence
    
    1. 복원 손실 (Reconstruction Loss):
       - 원본과 복원 이미지 간 차이 (MSE)
       - 이미지를 잘 복원하도록 유도
    
    2. KL Divergence:
       - 잠재 분포 N(μ, σ²)와 표준정규분포 N(0,1) 간 거리
       - 잠재 공간을 정규화하여 부드럽게 만듦
       - 과적합 방지 및 생성 능력 향상
    """
    recon_loss = F.mse_loss(recon, x, reduction='sum')  # 복원 손실
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence
    return recon_loss + kld

# 학습 루프
for epoch in range(10):  # 빠른 테스트용 10 epoch (실전: 50~200 epoch 권장)
    total_loss = 0
    for x, _ in loader:  # 배치 단위 학습
        recon, mu, logvar = model(x)  # Forward: 이미지 → 복원 이미지 + 분포 파라미터
        loss = loss_function(recon, x, mu, logvar)  # 손실 계산

        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()        # 역전파
        optimizer.step()       # 파라미터 업데이트

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss = {total_loss:.2f}")


import matplotlib.pyplot as plt
import numpy as np

def Test_image():
    """노이즈 제거 테스트: 10개 이미지에 노이즈 추가 후 VAE로 복원"""
    model.eval()  # 평가 모드 (Dropout 등 비활성화)
    
    # 결과 저장 디렉토리 생성
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 데이터셋에서 무작위로 10개 이미지 선택
    import random
    indices = random.sample(range(len(dataset)), 10)
    
    # 3행 × 10열 그리드 생성 (원본, 노이즈, 복원)
    fig, axs = plt.subplots(3, 10, figsize=(20, 6))
    
    for i, idx in enumerate(indices):
        img, _ = dataset[idx]        # 원본 이미지 로드
        img = img.unsqueeze(0)       # 배치 차원 추가: (3,64,64) → (1,3,64,64)

        # 가우시안 노이즈 추가 (표준편차 0.3)
        noise = 0.3 * torch.randn_like(img)
        noisy_img = torch.clamp(img + noise, 0, 1)  # 0~1 범위 유지

        # VAE로 노이즈 제거
        with torch.no_grad():  # 기울기 계산 비활성화 (메모리 절약)
            recon, _, _ = model(noisy_img)

        # ========== 시각화 ==========
        # 1행: 원본 이미지
        axs[0, i].imshow(np.transpose(img.squeeze().numpy(), (1, 2, 0)))
        axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_title("Original", fontsize=10)
        
        # 2행: 노이즈 추가된 이미지
        axs[1, i].imshow(np.transpose(noisy_img.squeeze().numpy(), (1, 2, 0)))
        axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_title("Noisy", fontsize=10)
        
        # 3행: VAE로 복원된 이미지 (노이즈 제거됨)
        axs[2, i].imshow(np.transpose(recon.squeeze().numpy(), (1, 2, 0)))
        axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_title("Denoised", fontsize=10)
    
    plt.tight_layout()
    
    # 결과 이미지 저장
    save_path = os.path.join(RESULT_DIR, "vae_denoising_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"결과 이미지 저장됨: {save_path}")
    
    # 화면에 3초간 표시 후 닫기
    plt.show(block=False)
    plt.pause(3)
    plt.close()


Test_image()
