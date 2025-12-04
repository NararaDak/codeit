import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filelock import FileLock
import os
import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ════════════════════════════════════════
# ▣ Meta/유틸리티함수.
# ════════════════════════════════════════
ver = "2025.12.01.002"
#BASE_DIR = r"D:\01.project\CodeIt\mission8\data"
#BASE_DIR = "/content/drive/MyDrive/codeit/mission8/data"
#BASE_DIR = r"d:\01.project\codeitmission8\mission8\data"
BASE_DIR = "/content/drive/MyDrive/project/data"
#BASE_DIR = r"D:\01.project\CodeIt\data"
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
        print(f"Log write error: {e}")

## GAN 평가 지표 CSV 저장 함수
def save_metrics_to_csv(model_name, transform_type, epoch_index, max_epochs, 
                        d_loss, g_loss, d_real_acc, d_fake_acc, current_lr,
                        val_d_loss=None, val_g_loss=None, val_d_real_acc=None, val_d_fake_acc=None):
    """GAN 학습 메트릭을 CSV 파일에 저장
    
    Args:
        model_name: 모델 이름 (conditional, advanced 등)
        transform_type: Transform 타입 (A, B, C)
        epoch_index: 현재 에포크 (1-based)
        max_epochs: 최대 에포크
        d_loss: Train Discriminator Loss
        g_loss: Train Generator Loss
        d_real_acc: Train Discriminator Real Accuracy
        d_fake_acc: Train Discriminator Fake Accuracy
        current_lr: 현재 학습률
        val_d_loss: Validation Discriminator Loss (optional)
        val_g_loss: Validation Generator Loss (optional)
        val_d_real_acc: Validation Real Accuracy (optional)
        val_d_fake_acc: Validation Fake Accuracy (optional)
    """
    new_data = {
        'timestamp': [now_str()],
        'Model': [model_name],
        'Transform': [transform_type],
        'Max_Epochs': [max_epochs],
        'Epoch': [epoch_index],
        'Train_D_Loss': [round(d_loss, 6)],
        'Train_G_Loss': [round(g_loss, 6)],
        'Train_D_Real_Acc': [round(d_real_acc, 4)],
        'Train_D_Fake_Acc': [round(d_fake_acc, 4)],
        'Val_D_Loss': [round(val_d_loss, 6) if val_d_loss is not None else None],
        'Val_G_Loss': [round(val_g_loss, 6) if val_g_loss is not None else None],
        'Val_D_Real_Acc': [round(val_d_real_acc, 4) if val_d_real_acc is not None else None],
        'Val_D_Fake_Acc': [round(val_d_fake_acc, 4) if val_d_fake_acc is not None else None],
        'Learning_Rate': [round(current_lr, 8)]
    }
    
    filename = RESULT_CSV
    lock_filename = filename + ".lock"
    new_df = pd.DataFrame(new_data)
    
    try:
        makedirs(os.path.dirname(filename))
        lock = FileLock(lock_filename, timeout=10)
        with lock:
            if os.path.exists(filename):
                try:
                    existing_df = pd.read_csv(filename)
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                    updated_df.to_csv(filename, index=False)
                except:
                    new_df.to_csv(filename, index=False)
            else:
                new_df.to_csv(filename, index=False)
    except Exception as e:
        print(f"CSV 저장 중 오류 발생: {e}")
        OpLog(f"Error saving CSV: {e}")

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  # 부하 감소: 64→32
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
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
        
        # Early Stopping 및 Learning Rate Scheduler 관련 변수
        self._scheduler_G = None
        self._scheduler_D = None
        self._best_d_loss = float('inf')
        self._patience_counter = 0

    def _evaluate_model(self, data_loader):
        """Validation/Test 데이터로 모델 평가
        
        Args:
            data_loader: 평가용 데이터 로더
            
        Returns:
            tuple: (avg_d_loss, avg_g_loss, d_real_acc, d_fake_acc)
        """
        self._generator.eval()
        self._discriminator.eval()
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_d_real_correct = 0
        total_d_fake_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for imgs, labels in data_loader:
                batch_size = imgs.size(0)
                imgs = imgs.to(MY_META.device())
                labels = labels.to(MY_META.device())
                
                valid = torch.ones(batch_size, 1, device=MY_META.device())
                fake = torch.zeros(batch_size, 1, device=MY_META.device())
                
                # Discriminator loss 계산
                real_loss = self._criterion(self._discriminator(imgs, labels), valid)
                
                noise = torch.randn(batch_size, self._latent_dim, device=MY_META.device())
                gen_imgs = self._generator(noise, labels)
                fake_loss = self._criterion(self._discriminator(gen_imgs, labels), fake)
                d_loss = real_loss + fake_loss
                
                # Generator loss 계산
                g_loss = self._criterion(self._discriminator(gen_imgs, labels), valid)
                
                # Accuracy 계산
                d_real_pred = self._discriminator(imgs, labels)
                d_fake_pred = self._discriminator(gen_imgs, labels)
                d_real_correct = (d_real_pred > 0.5).sum().item()
                d_fake_correct = (d_fake_pred < 0.5).sum().item()
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                total_d_real_correct += d_real_correct
                total_d_fake_correct += d_fake_correct
                total_samples += batch_size
        
        # 평균 계산
        num_batches = len(data_loader)
        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        d_real_acc = total_d_real_correct / total_samples
        d_fake_acc = total_d_fake_correct / total_samples
        
        self._generator.train()
        self._discriminator.train()
        
        return avg_d_loss, avg_g_loss, d_real_acc, d_fake_acc

    def fit(self, trans_type, trainLoader, testLoader, epochs=50, lr=0.0002, betas=(0.5, 0.999)):
        # (patience 인자 제거)
        self._trans_type = trans_type
        self._idx_to_class = {i: class_name for i, class_name in enumerate(trainLoader.dataset.classes)}
        self._train_loader = trainLoader
        self._test_loader = testLoader
        self._epochs = epochs
        self._lr = lr
        
        self._optimizer_G = optim.Adam(self._generator.parameters(), lr=lr, betas=betas)
        self._optimizer_D = optim.Adam(self._discriminator.parameters(), lr=lr, betas=betas)
        
        model_name = self.__class__.__name__.replace('GanModel', '').replace('Model', '')

        Lines(f"[{model_name}] 학습 시작 - Early Stopping 비활성화됨")
        for epoch in range(epochs):
            epoch_d_loss_sum = 0.0
            epoch_g_loss_sum = 0.0
            
            for i, (imgs, labels) in enumerate(self._train_loader):
                batch_size_current = imgs.size(0)
                imgs = imgs.to(MY_META.device())
                labels = labels.to(MY_META.device())

                # [수정 1] Label Smoothing: Real Label을 1.0 대신 0.9로 설정
                valid = torch.full((batch_size_current, 1), 0.9, device=MY_META.device())
                fake  = torch.zeros(batch_size_current, 1, device=MY_META.device())

                # --- 판별자 학습 ---
                self._optimizer_D.zero_grad()
                
                # 진짜 이미지 판별
                real_loss = self._criterion(self._discriminator(imgs, labels), valid)
                
                # 가짜 이미지 판별
                noise = torch.randn(batch_size_current, self._latent_dim, device=MY_META.device())
                gen_imgs = self._generator(noise, labels)
                
                # .detach()를 사용하여 Generator로의 역전파 차단
                fake_loss = self._criterion(self._discriminator(gen_imgs.detach(), labels), fake)
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self._optimizer_D.step()

                # --- 생성자 학습 ---
                self._optimizer_G.zero_grad()
                
                # 생성자는 판별자가 가짜를 '1(진짜)'로 인식하게 만들어야 함 (Non-saturating loss)
                # 생성자 학습 시에는 Label Smoothing을 쓰지 않고 1.0을 목표로 하는 것이 일반적입니다.
                valid_g = torch.ones(batch_size_current, 1, device=MY_META.device())
                g_loss = self._criterion(self._discriminator(gen_imgs, labels), valid_g)
                
                g_loss.backward()
                self._optimizer_G.step()
                
                epoch_d_loss_sum += d_loss.item()
                epoch_g_loss_sum += g_loss.item()
                
                print(f"\r[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(self._train_loader)}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}", end="")
            
            # 평균 계산
            train_avg_d_loss = epoch_d_loss_sum / len(self._train_loader)
            train_avg_g_loss = epoch_g_loss_sum / len(self._train_loader)
            
            print(f"\n[Epoch {epoch+1} 완료] Avg D_loss: {train_avg_d_loss:.4f}, Avg G_loss: {train_avg_g_loss:.4f}")

            # [수정 2] 시각화 및 무조건 저장 (Early Stopping 로직 제거)
            self._visualize_results(epoch, epochs)
            
            # 마지막 에포크이거나 10 에포크마다 저장
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                self.save_model(epoch + 1)

    def fit_org(self, trans_type, trainLoader, testLoader, epochs=50, lr=0.0002, betas=(0.5, 0.999), patience=5):
        """GAN 모델 학습 (Early Stopping + LR Scheduler 적용)
        
        Args:
            epochs: 최대 에포크 (기본 50)
            patience: Early Stopping patience (기본 5)
            lr: 초기 학습률 (0.0002 → 30 epoch까지 0.00005로 감소)
        """
        self._trans_type = trans_type
        self._idx_to_class = {i: class_name for i, class_name in enumerate(trainLoader.dataset.classes)}
        self._train_loader = trainLoader
        self._test_loader = testLoader
        self._epochs = epochs
        self._lr = lr
        self._betas = betas
        self._optimizer_G = optim.Adam(self._generator.parameters(), lr=lr, betas=betas)
        self._optimizer_D = optim.Adam(self._discriminator.parameters(), lr=lr, betas=betas)
        
        # Learning Rate Scheduler: 30 epoch까지 0.0002 → 0.00005로 감소
        # 30 이후는 0.00005 고정
        lambda_lr = lambda epoch: max(0.25, 1.0 - epoch / 30 * 0.75) if epoch < 30 else 0.25
        self._scheduler_G = optim.lr_scheduler.LambdaLR(self._optimizer_G, lr_lambda=lambda_lr)
        self._scheduler_D = optim.lr_scheduler.LambdaLR(self._optimizer_D, lr_lambda=lambda_lr)

        # 모델 이름 추출 (클래스명에서)
        model_name = self.__class__.__name__.replace('GanModel', '').replace('Model', '')

        for epoch in range(epochs):
            epoch_d_loss_sum = 0.0
            epoch_g_loss_sum = 0.0
            epoch_d_real_correct = 0
            epoch_d_fake_correct = 0
            epoch_total_samples = 0
            
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
                
                # Discriminator 정확도 계산
                d_real_pred = self._discriminator(imgs, labels)
                d_fake_pred = self._discriminator(gen_imgs.detach(), labels)
                d_real_correct = (d_real_pred > 0.5).sum().item()
                d_fake_correct = (d_fake_pred < 0.5).sum().item()

                # 생성자 학습
                self._optimizer_G.zero_grad()
                g_loss = self._criterion(self._discriminator(gen_imgs, labels), valid)
                g_loss.backward()
                self._optimizer_G.step()
                
                # 에포크 메트릭 누적
                epoch_d_loss_sum += d_loss.item()
                epoch_g_loss_sum += g_loss.item()
                epoch_d_real_correct += d_real_correct
                epoch_d_fake_correct += d_fake_correct
                epoch_total_samples += batch_size_current
                
                msg = f"[{model_name}/{self._trans_type}] [Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(self._train_loader)}] "
                msg += f"D_loss: {d_loss.item():.4f}  G_loss : {g_loss.item():.4f}"
                OpLog(msg, bLines=False)
                print(msg, end="\r")
            
            # 에포크 종료 후 Train 평균 메트릭 계산
            num_batches = len(self._train_loader)
            train_avg_d_loss = epoch_d_loss_sum / num_batches
            train_avg_g_loss = epoch_g_loss_sum / num_batches
            train_d_real_acc = epoch_d_real_correct / epoch_total_samples
            train_d_fake_acc = epoch_d_fake_correct / epoch_total_samples
            current_lr = self._optimizer_G.param_groups[0]['lr']
            
            # Validation 평가
            val_avg_d_loss, val_avg_g_loss, val_d_real_acc, val_d_fake_acc = self._evaluate_model(self._test_loader)
            
            # CSV에 메트릭 저장 (Train + Val)
            save_metrics_to_csv(
                model_name=model_name,
                transform_type=self._trans_type,
                epoch_index=epoch + 1,
                max_epochs=epochs,
                d_loss=train_avg_d_loss,
                g_loss=train_avg_g_loss,
                d_real_acc=train_d_real_acc,
                d_fake_acc=train_d_fake_acc,
                current_lr=current_lr,
                val_d_loss=val_avg_d_loss,
                val_g_loss=val_avg_g_loss,
                val_d_real_acc=val_d_real_acc,
                val_d_fake_acc=val_d_fake_acc
            )
            
            print(f"\n[Epoch {epoch+1}/{epochs}]")
            print(f"  Train - D_loss: {train_avg_d_loss:.4f}, G_loss: {train_avg_g_loss:.4f}, "
                  f"D_Real_Acc: {train_d_real_acc:.2%}, D_Fake_Acc: {train_d_fake_acc:.2%}")
            print(f"  Val   - D_loss: {val_avg_d_loss:.4f}, G_loss: {val_avg_g_loss:.4f}, "
                  f"D_Real_Acc: {val_d_real_acc:.2%}, D_Fake_Acc: {val_d_fake_acc:.2%}")
            print(f"  LR: {current_lr:.6f}")
            
            # Learning Rate Scheduler 업데이트
            self._scheduler_G.step()
            self._scheduler_D.step()
            
            # Early Stopping 체크 (Validation D_loss 기준)
            if val_avg_d_loss < self._best_d_loss:
                self._best_d_loss = val_avg_d_loss
                self._patience_counter = 0
                print(f"  ✓ Best Val D_loss 갱신: {self._best_d_loss:.4f}")
                # Best 모델 저장
                self.save_model(epoch + 1, is_best=True)
            else:
                self._patience_counter += 1
                print(f"  ⚠ {self._patience_counter}/{patience} - Val D_loss 개선 없음 (현재: {val_avg_d_loss:.4f} vs 최고: {self._best_d_loss:.4f})")
                
                if self._patience_counter >= patience:
                    print(f"\n[Early Stopping] {patience} epochs 동안 개선 없음. 학습 종료.")
                    break

            # 에포크마다 생성 이미지 시각화 (각 클래스별로 3개씩 출력)
            #if (epoch + 1) % 10 == 0:
            self._visualize_results(epoch, epochs)

    def _visualize_results(self, epoch, max_epochs):
        """생성 이미지 시각화"""
        self._generator.eval()
        n_row = 3               # 각 클래스당 3개씩
        n_col = self._num_classes     # 총 10개 클래스
        total_samples = n_row * n_col
        noise = torch.randn(total_samples, self._latent_dim, device=MY_META.device())
        labels_sample = torch.arange(0, self._num_classes, device=MY_META.device()).repeat(n_row)
        gen_imgs = self._generator(noise, labels_sample).detach().cpu()
        
        # [-1, 1] 범위를 [0, 1]로 변환
        gen_imgs = (gen_imgs + 1) / 2.0
        gen_imgs = torch.clamp(gen_imgs, 0, 1)

        fig, axs = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
        for i in range(n_row):
            for j in range(n_col):
                idx = i * n_col + j
                axs[i, j].imshow(gen_imgs[idx, 0, :, :], cmap='gray', vmin=0, vmax=1)
                axs[i, j].axis('off')
                if i == 0:
                    axs[i, j].set_title(self._idx_to_class[j], fontsize=10)
        
        plt.tight_layout()
        
        # 저장을 먼저 하고, 그 다음에 표시/닫기
        result_dir = f"{BASE_DIR}/model_results"
        makedirs(result_dir)
        filename = f"{result_dir}/{self.__class__.__name__}_{self._trans_type}_epoch{epoch+1}_of_{max_epochs}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"생성 이미지 저장됨: {filename}")
        
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        self._generator.train()

    ## 모델 저장 함수 
    def save_model(self, epoch_index, is_best=False):
        """현재 모델 상태를 저장
        
        Args:
            epoch_index: 현재 에포크 번호
            is_best: Best 모델인지 여부 (파일명에 'best' 표시)
        """
        save_dir = f"{BASE_DIR}/modelfiles"
        makedirs(save_dir)
        model_name = self.__class__.__name__
        
        # 실제 적용된 현재 학습률 가져오기
        current_lr = self._optimizer_G.param_groups[0]['lr']
        
        # Best 모델은 파일명에 'best' 표시
        best_tag = "_BEST" if is_best else ""
        filename = f"{save_dir}/{model_name}_{self._trans_type}{best_tag}_ep{epoch_index}_lr{current_lr:.6f}.pth"
        
        torch.save({
            'epoch': epoch_index,
            'generator_state_dict': self._generator.state_dict(),
            'discriminator_state_dict': self._discriminator.state_dict(),
            'optimizer_G_state_dict': self._optimizer_G.state_dict(),
            'optimizer_D_state_dict': self._optimizer_D.state_dict(),
            'best_d_loss': self._best_d_loss,
            'current_lr': current_lr,  # 실제 학습률도 저장
            'initial_lr': self._lr,    # 초기 학습률 참고용
            'is_best': is_best,        # Best 모델 여부
        }, filename)
        
        if is_best:
            print(f"  🏆 Best 모델 저장됨: {filename}")
            OpLog(f"Best model saved: {filename}")
        else:
            print(f"  모델 체크포인트 저장됨: {filename}")
            OpLog(f"Checkpoint saved: {filename}")

    ## 모델 로드 함수
    def load_model(self, model_file):
        """저장된 모델 상태를 로드
        
        Args:
            model_file: 모델 파일 경로
        """
        checkpoint = torch.load(model_file, map_location=MY_META.device())
        self._generator.load_state_dict(checkpoint['generator_state_dict'])
        self._discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if self._optimizer_G is not None:
            self._optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        if self._optimizer_D is not None:
            self._optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self._best_d_loss = checkpoint.get('best_d_loss', float('inf'))
        self._generator.eval()
        self._discriminator.eval()
        print(f"모델 로드됨: {model_file}, Epoch: {checkpoint['epoch']}")
        OpLog(f"Model loaded: {model_file}")
        return checkpoint['epoch']

class ConditionalGanModel(BaseGan):
    # 생성자 (Generator)
    class Generator(nn.Module):
        def __init__(self, latent_dim, num_classes, image_size):
            super(ConditionalGanModel.Generator, self).__init__()
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
            super(ConditionalGanModel.Discriminator, self).__init__()
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
        super(ConditionalGanModel, self).__init__(latent_dim)
        # Generator와 Discriminator 생성 시 파라미터 전달
        self._generator = self.Generator(self._latent_dim, self._num_classes, self._image_size).to(MY_META.device())
        self._discriminator = self.Discriminator(self._num_classes, self._image_size).to(MY_META.device())

class AdvancedGanModel(BaseGan):
    # ─────────────────────────────
    # Self-Attention Layer (SAGAN)
    # ─────────────────────────────
    class SelfAttention(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
            self.key   = nn.Conv2d(in_dim, in_dim // 8, 1)
            self.value = nn.Conv2d(in_dim, in_dim, 1)
            self.gamma = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            B, C, H, W = x.size()

            query = self.query(x).view(B, -1, H * W)          # B, C/8, HW
            key   = self.key(x).view(B, -1, H * W)            # B, C/8, HW
            energy = torch.bmm(query.permute(0, 2, 1), key)   # B, HW, HW
            attention = torch.softmax(energy, dim=-1)

            value = self.value(x).view(B, C, H * W)           # B, C, HW
            out = torch.bmm(value, attention.permute(0, 2, 1))
            out = out.view(B, C, H, W)

            out = self.gamma * out + x
            return out

    # ─────────────────────────────
    # Residual Block (Generator 용)
    # ─────────────────────────────
    class ResBlockUp(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.upsample = nn.Upsample(scale_factor=2)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, x):
            shortcut = self.upsample(self.shortcut(x))

            x = self.upsample(x)
            x = self.bn1(self.conv1(x))
            x = F.relu(x)
            x = self.bn2(self.conv2(x))

            return F.relu(x + shortcut)

    # ─────────────────────────────
    # Residual Block (Discriminator 용)
    # ─────────────────────────────
    class ResBlockDown(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            )
            self.conv2 = nn.utils.spectral_norm(
                nn.Conv2d(out_channels, out_channels, 3, 2, 1)
            )
            self.shortcut = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1, stride=2)
            )

        def forward(self, x):
            shortcut = self.shortcut(x)

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))

            return x + shortcut

    # ─────────────────────────────
    # Generator
    # ─────────────────────────────
    class Generator(nn.Module):
        def __init__(self, latent_dim, num_classes, img_size):
            super().__init__()
            self.latent_dim = latent_dim
            self.num_classes = num_classes
            self.img_size = img_size

            self.label_emb = nn.Embedding(num_classes, num_classes)

            init_size = img_size // 4  # 7
            self.fc = nn.Sequential(
                nn.Linear(latent_dim + num_classes, 256 * init_size * init_size),
                nn.ReLU(True)
            )

            self.block1 = AdvancedGanModel.ResBlockUp(256, 128)
            self.attn = AdvancedGanModel.SelfAttention(128)
            self.block2 = AdvancedGanModel.ResBlockUp(128, 64)

            self.output = nn.Sequential(
                nn.Conv2d(64, 1, 3, 1, 1),
                nn.Tanh()
            )

        def forward(self, z, labels):
            label = self.label_emb(labels)
            gen_in = torch.cat((z, label), dim=1)

            out = self.fc(gen_in)
            out = out.view(out.size(0), 256, self.img_size // 4, self.img_size // 4)

            out = self.block1(out)
            out = self.attn(out)
            out = self.block2(out)

            img = self.output(out)
            return img

    # ─────────────────────────────
    # Discriminator
    # ─────────────────────────────
    class Discriminator(nn.Module):
        def __init__(self, num_classes, img_size):
            super().__init__()
            self.num_classes = num_classes
            self.img_size = img_size

            self.label_emb = nn.Embedding(num_classes, 1)

            self.block1 = AdvancedGanModel.ResBlockDown(1 + 1, 64)
            self.attn = AdvancedGanModel.SelfAttention(64)
            self.block2 = AdvancedGanModel.ResBlockDown(64, 128)

            final_size = img_size // 4  # 7
            self.fc = nn.utils.spectral_norm(
                nn.Linear(128 * final_size * final_size, 1)
            )

        def forward(self, img, labels):
            B = img.size(0)
            label = self.label_emb(labels).view(B, 1, 1, 1).expand(B, 1, self.img_size, self.img_size)

            x = torch.cat((img, label), dim=1)

            x = self.block1(x)
            x = self.attn(x)
            x = self.block2(x)

            x = x.view(B, -1)
            validity = torch.sigmoid(self.fc(x))
            return validity

    # ─────────────────────────────
    # Model 초기화
    # ─────────────────────────────
    def __init__(self, latent_dim=100):
        super().__init__(latent_dim)
        self._generator = AdvancedGanModel.Generator(latent_dim, self._num_classes, self._image_size).to(MY_META.device())
        self._discriminator = AdvancedGanModel.Discriminator(self._num_classes, self._image_size).to(MY_META.device())


class StyleGAN2Model(BaseGan):
    """
    StyleGAN2-ADA 기반 GAN 모델
    - Style-based Generator with Adaptive Discriminator Augmentation
    - Mapping Network: latent code → intermediate latent space (W)
    - Synthesis Network: W-space → 이미지 생성
    - Progressive Growing 없이 안정적 학습
    """
    
    # ─────────────────────────────
    # Style Modulation Block
    # ─────────────────────────────
    class StyleModulationLayer(nn.Module):
        """스타일 변조 레이어 (AdaIN)"""
        def __init__(self, latent_dim, num_features):
            super().__init__()
            self.style_scale = nn.Linear(latent_dim, num_features)
            self.style_bias = nn.Linear(latent_dim, num_features)
            
        def forward(self, x, w):
            # w: (B, latent_dim) - intermediate latent code
            # x: (B, C, H, W) - feature map
            scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
            
            # Normalize feature map
            x_norm = F.instance_norm(x)
            
            # Apply style modulation
            return scale * x_norm + bias
    
    # ─────────────────────────────
    # Synthesis Block (Conv + Style Modulation)
    # ─────────────────────────────
    class SynthesisBlock(nn.Module):
        def __init__(self, in_channels, out_channels, latent_dim, upsample=True):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=2) if upsample else None
            
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.style_mod1 = StyleGAN2Model.StyleModulationLayer(latent_dim, out_channels)
            self.noise1 = nn.Parameter(torch.randn(1, 1, 1, 1))
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.style_mod2 = StyleGAN2Model.StyleModulationLayer(latent_dim, out_channels)
            self.noise2 = nn.Parameter(torch.randn(1, 1, 1, 1))
            
        def forward(self, x, w):
            if self.upsample:
                x = self.upsample(x)
            
            # First convolution + style modulation
            x = self.conv1(x)
            x = self.style_mod1(x, w)
            x = F.leaky_relu(x + self.noise1, 0.2)
            
            # Second convolution + style modulation
            x = self.conv2(x)
            x = self.style_mod2(x, w)
            x = F.leaky_relu(x + self.noise2, 0.2)
            
            return x
    
    # ─────────────────────────────
    # Mapping Network (Z → W)
    # ─────────────────────────────
    class MappingNetwork(nn.Module):
        def __init__(self, latent_dim, num_classes, hidden_dim=256, num_layers=4):
            super().__init__()
            self.latent_dim = latent_dim
            self.num_classes = num_classes
            
            # Class embedding
            self.label_emb = nn.Embedding(num_classes, latent_dim)
            
            # Mapping network: Z + label → W
            layers = []
            layers.append(nn.Linear(latent_dim * 2, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            
            self.mapping = nn.Sequential(*layers)
            
        def forward(self, z, labels):
            # z: (B, latent_dim)
            # labels: (B,)
            label_emb = self.label_emb(labels)  # (B, latent_dim)
            x = torch.cat([z, label_emb], dim=1)  # (B, latent_dim * 2)
            w = self.mapping(x)  # (B, hidden_dim) = W-space
            return w
    
    # ─────────────────────────────
    # Synthesis Network (W → Image)
    # ─────────────────────────────
    class SynthesisNetwork(nn.Module):
        def __init__(self, latent_dim, img_size):
            super().__init__()
            self.latent_dim = latent_dim
            self.img_size = img_size
            
            # Constant input (학습 가능한 시작점)
            init_size = img_size // 4  # 7
            self.const_input = nn.Parameter(torch.randn(1, 256, init_size, init_size))
            
            # Synthesis blocks
            self.block1 = StyleGAN2Model.SynthesisBlock(256, 128, latent_dim, upsample=True)   # 7 → 14
            self.block2 = StyleGAN2Model.SynthesisBlock(128, 64, latent_dim, upsample=True)    # 14 → 28
            
            # To RGB
            self.to_rgb = nn.Conv2d(64, 1, 1)
            
        def forward(self, w):
            # w: (B, latent_dim)
            B = w.size(0)
            
            # Start from constant
            x = self.const_input.repeat(B, 1, 1, 1)
            
            # Apply synthesis blocks
            x = self.block1(x, w)
            x = self.block2(x, w)
            
            # Convert to image
            img = torch.tanh(self.to_rgb(x))
            return img
    
    # ─────────────────────────────
    # Generator (Mapping + Synthesis)
    # ─────────────────────────────
    class Generator(nn.Module):
        def __init__(self, latent_dim, num_classes, img_size):
            super().__init__()
            self.mapping_network = StyleGAN2Model.MappingNetwork(latent_dim, num_classes, hidden_dim=256)
            self.synthesis_network = StyleGAN2Model.SynthesisNetwork(latent_dim=256, img_size=img_size)
            
        def forward(self, z, labels):
            w = self.mapping_network(z, labels)
            img = self.synthesis_network(w)
            return img
    
    # ─────────────────────────────
    # Discriminator (with ADA - Adaptive Augmentation)
    # ─────────────────────────────
    class Discriminator(nn.Module):
        def __init__(self, num_classes, img_size):
            super().__init__()
            self.num_classes = num_classes
            self.img_size = img_size
            
            # Label embedding
            self.label_emb = nn.Embedding(num_classes, 1)
            
            # Discriminator blocks with Spectral Normalization
            self.block1 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(1 + 1, 64, 3, 2, 1)),   # 28 → 14
                nn.LeakyReLU(0.2)
            )
            
            self.block2 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)),     # 14 → 7
                nn.LeakyReLU(0.2)
            )
            
            self.block3 = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1)),
                nn.LeakyReLU(0.2)
            )
            
            final_size = img_size // 4
            self.fc = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(256 * final_size * final_size, 1)),
                nn.Sigmoid()
            )
            
        def forward(self, img, labels):
            B = img.size(0)
            label = self.label_emb(labels).view(B, 1, 1, 1).expand(B, 1, self.img_size, self.img_size)
            
            # Concatenate image and label
            x = torch.cat([img, label], dim=1)
            
            # Forward through blocks
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            
            # Flatten and classify
            x = x.view(B, -1)
            validity = self.fc(x)
            return validity
    
    # ─────────────────────────────
    # Model 초기화
    # ─────────────────────────────
    def __init__(self, latent_dim=100):
        super().__init__(latent_dim)
        self._generator = StyleGAN2Model.Generator(latent_dim, self._num_classes, self._image_size).to(MY_META.device())
        self._discriminator = StyleGAN2Model.Discriminator(self._num_classes, self._image_size).to(MY_META.device())

        
# 모델 초기화 및 학습 실행
# model = ConditionalGanModel()
# model.fit(epochs=30)

def Single_Train(Wahtmodel="conditional", trans_type='A', epochs=50, lr=0.0002, betas=(0.5, 0.999), patience=5):
    """GAN 모델 학습 (Early Stopping + LR Scheduler 적용)
    
    Args:
        Wahtmodel: 모델 선택 ('conditional', 'advanced', 'stylegan2')
        epochs: 최대 에포크 (기본 50)
        patience: Early Stopping patience (기본 5)
        lr: 초기 학습률 (0.0002 → 30 epoch까지 0.00005로 감소)
    """
    if Wahtmodel == "conditional":
        model = ConditionalGanModel()
    elif Wahtmodel == "advanced":
        model = AdvancedGanModel()
    elif Wahtmodel == "stylegan2":
        model = StyleGAN2Model()
    else:
        raise ValueError(f"Unknown Wahtmodel: {Wahtmodel}. Choose 'conditional', 'advanced', or 'stylegan2'.")
    train_loader, test_loader = GetLoader(transform_type=trans_type)
    model.fit(trans_type=trans_type, trainLoader=train_loader, testLoader=test_loader, 
              epochs=epochs, lr=lr, betas=betas, patience=patience)

def Multi_train(Wahtmodel="conditional", epochs=50, lr=0.0002, betas=(0.5, 0.999), patience=5):
    for trans_type in ["A", "B", "C"]:
        Single_Train(Wahtmodel=Wahtmodel, trans_type=trans_type, epochs=epochs, lr=lr, betas=betas, patience=patience)
    

Multi_train(Wahtmodel="conditional", epochs=50, lr=0.0002, betas=(0.5, 0.999), patience=5)
#Multi_train(Wahtmodel="advanced", epochs=50, lr=0.0002, betas=(0.5, 0.999), patience=5)
#Multi_train(Wahtmodel="stylegan2", epochs=50, lr=0.0002, betas=(0.5, 0.999), patience=5)


