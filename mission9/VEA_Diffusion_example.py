import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import os
import math
import matplotlib.pyplot as plt
import numpy as np

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. 데이터셋 설정 (기존과 동일)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Diffusion은 -1~1 범위 권장
])

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                          glob.glob(os.path.join(image_dir, "*.png")) + \
                          glob.glob(os.path.join(image_dir, "*.jpeg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 64, 64), 0

# 경로 설정 (사용자 경로에 맞게 수정 필요)
IMG_DIR = r"D:\01.project\CodeIt\data\catanddog\cats"
RESULT_DIR = r"D:\01.project\CodeIt\data\catanddog\result"
os.makedirs(RESULT_DIR, exist_ok=True)

dataset = ImageDataset(image_dir=IMG_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

# ==========================================
# 2. Base Model (공통 학습 로직)
# ==========================================
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optimizer = None

    def configure_optimizer(self, lr=1e-3):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def training_step(self, batch):
        """자식 클래스에서 구체적인 손실 계산 로직 구현"""
        raise NotImplementedError

    def fit(self, dataloader, epochs=10):
        self.to(device)
        self.train()
        
        loss_history = []
        for epoch in range(epochs):
            total_loss = 0
            for i, (x, _) in enumerate(dataloader):
                x = x.to(device)
                
                # 자식 클래스의 training_step 호출
                loss = self.training_step(x)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            loss_history.append(avg_loss)
            print(f"[{self.__class__.__name__}] Epoch {epoch+1}/{epochs}, Loss = {avg_loss:.4f}")
        return loss_history

# ==========================================
# 3. VAE Implementation
# ==========================================
class VAE(BaseModel):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), 
            nn.Tanh() # Normalization(-1~1)에 맞춰 Tanh 사용
        )
        self.configure_optimizer()

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, x):
        recon, mu, logvar = self.forward(x)
        # Loss: MSE + KL Divergence
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kld) / x.size(0) # 배치 크기로 나누어 평균화

# ==========================================
# 4. Diffusion Implementation (DDPM)
# ==========================================

# --- Diffusion용 Helper Block: Time Embedding ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# --- Diffusion용 Helper Block: Simple U-Net Block ---
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # 1. First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 2. Time Embedding Injection
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2] # (B, C, 1, 1)로 확장
        h = h + time_emb
        # 3. Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 4. Down/Up Sample
        return self.transform(h)

class DiffusionUNet(BaseModel):
    def __init__(self, img_size=64, T=1000):
        super().__init__()
        self.img_size = img_size
        self.T = T # 총 Time Step

        # === Pre-calculate Diffusion Hyperparameters (Beta Schedule) ===
        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # 누적 곱

        # === U-Net Architecture ===
        # Time Embedding
        time_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Down Path
        self.down1 = Block(3, 64, time_dim)         # 64 -> 32
        self.down2 = Block(64, 128, time_dim)       # 32 -> 16
        self.down3 = Block(128, 256, time_dim)      # 16 -> 8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU()
        )

        # Up Path (Concatenation을 위해 채널 수 조정)
        # Skip connection concat 후 채널: 512+256=768, 256+128=384, 128+64=192
        # Block의 up=True는 입력을 2배로 기대하므로 절반만 전달
        self.up1 = Block(768 // 2, 256, time_dim, up=True) # 384 입력 (2배=768) -> 256, 8 -> 16
        self.up2 = Block(384 // 2, 128, time_dim, up=True) # 192 입력 (2배=384) -> 128, 16 -> 32
        self.up3 = Block(192 // 2, 64, time_dim, up=True)  # 96 입력 (2배=192) -> 64, 32 -> 64
        
        self.final_conv = nn.Conv2d(64, 3, 1) # Output Noise 예측

        self.configure_optimizer(lr=3e-4) # Diffusion은 보통 LR이 더 낮음

    def get_loss(self, x, t):
        # 1. Random Noise 생성
        noise = torch.randn_like(x)
        
        # 2. Forward Diffusion: q(x_t | x_0)
        # sqrt(alpha_hat) * x_0 + sqrt(1 - alpha_hat) * epsilon
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        
        # 3. Predict Noise: epsilon_theta(x_t, t)
        noise_pred = self.forward(x_noisy, t)
        
        # 4. Loss: 예측한 노이즈와 실제 노이즈 간의 MSE
        return F.mse_loss(noise_pred, noise)

    def forward(self, x, t):
        # Time Embed
        t_emb = self.time_mlp(t)
        
        # U-Net Down
        x1 = self.down1(x, t_emb) # Skip connection 저장
        x2 = self.down2(x1, t_emb)
        x3 = self.down3(x2, t_emb)
        
        # Bottleneck
        x_mid = self.bottleneck(x3)
        
        # U-Net Up (Concat with Skip connections)
        x = self.up1(torch.cat([x_mid, x3], dim=1), t_emb)
        x = self.up2(torch.cat([x, x2], dim=1), t_emb)
        x = self.up3(torch.cat([x, x1], dim=1), t_emb)
        
        return self.final_conv(x)

    def training_step(self, x):
        # 랜덤 Time Step 샘플링 (0 ~ T-1)
        t = torch.randint(0, self.T, (x.shape[0],), device=device).long()
        return self.get_loss(x, t)

    @torch.no_grad()
    def sample_image(self, num_images=10):
        """노이즈로부터 이미지 생성 (Reverse Process)"""
        self.eval()
        # 1. 완전한 노이즈에서 시작
        x = torch.randn((num_images, 3, self.img_size, self.img_size), device=device)
        
        # 2. T-1 부터 0까지 역순으로 노이즈 제거
        for i in reversed(range(self.T)):
            t = torch.full((num_images,), i, device=device, dtype=torch.long)
            predicted_noise = self.forward(x, t)
            
            # 수식: x_{t-1} 계산 (Sampler)
            alpha = self.alpha[i]
            alpha_hat = self.alpha_hat[i]
            beta = self.beta[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x) # 마지막 단계는 노이즈 없음
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
        self.train()
        # -1~1 범위를 0~1로 변환
        return (x.clamp(-1, 1) + 1) / 2

# ==========================================
# 5. 실행 및 결과 시각화
# ==========================================

def visualize_results(vae_model, diffusion_model, dataloader):
    vae_model.eval()
    diffusion_model.eval()
    
    # 1. 데이터 가져오기 (VAE 테스트용)
    real_imgs, _ = next(iter(dataloader))
    real_imgs = real_imgs[:10].to(device)
    
    # 2. VAE: Reconstruction (이미지 -> 압축 -> 복원)
    with torch.no_grad():
        recon_imgs, _, _ = vae_model(real_imgs)
        recon_imgs = (recon_imgs.clamp(-1, 1) + 1) / 2 # -1~1 -> 0~1
    
    # 3. Diffusion: Generation (노이즈 -> 생성)
    gen_imgs = diffusion_model.sample_image(num_images=10)
    
    # 시각화
    fig, axs = plt.subplots(3, 10, figsize=(20, 6))
    
    # Row 1: VAE Input (Real)
    for i in range(10):
        img = (real_imgs[i].cpu().permute(1, 2, 0).numpy() + 1) / 2
        axs[0, i].imshow(img)
        axs[0, i].axis('off')
        if i == 0: axs[0, i].set_title("Real (VAE Input)")

    # Row 2: VAE Output (Reconstruction)
    for i in range(10):
        img = recon_imgs[i].cpu().permute(1, 2, 0).numpy()
        axs[1, i].imshow(img)
        axs[1, i].axis('off')
        if i == 0: axs[1, i].set_title("VAE Recon")

    # Row 3: Diffusion Output (Generation)
    for i in range(10):
        img = gen_imgs[i].cpu().permute(1, 2, 0).numpy()
        axs[2, i].imshow(img)
        axs[2, i].axis('off')
        if i == 0: axs[2, i].set_title("Diffusion Generated")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "comparison_result.png"))
    plt.show()

# --- Main 실행 ---
if __name__ == "__main__":
    # 2. Diffusion 학습
    print("\n=== Training Diffusion ===")
    diffusion = DiffusionUNet(T=500) # 속도를 위해 T=500 설정
    diffusion.fit(loader, epochs=10) 

    # 1. VAE 학습
    print("=== Training VAE ===")
    vae = VAE()
    vae.fit(loader, epochs=10) # 테스트를 위해 Epoch 줄임
    
    # 3. 결과 비교
    print("\n=== Visualizing Results ===")
    visualize_results(vae, diffusion, loader)


