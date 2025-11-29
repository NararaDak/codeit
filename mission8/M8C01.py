import os
import random
import time
import datetime
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from filelock import FileLock

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.models as models
from torchvision import transforms as T
from torchvision.transforms import v2

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import typing  # 타입 힌트 사용을 위해 추가
    

# ════════════════════════════════════════
# ▣ Meta/유틸리티함수.
# ════════════════════════════════════════
ver = "2025.11.29.001"
#BASE_DIR = r"D:\01.project\CodeIt\mission8\data"
#BASE_DIR = "/content/drive/MyDrive/codeit/mission8/data"
#BASE_DIR = r"d:\01.project\codeitmission8\mission8\data"
BASE_DIR = r"D:\01.project\CodeIt\data"
LOG_FILE = f"{BASE_DIR}/m8log.txt"
RESULT_CSV = f"{BASE_DIR}/result.csv"
BASE_DIR = f"{BASE_DIR}/football"

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
        
    log_filename = os.path.join(BASE_DIR, "op_log.txt")
    log_lock_filename = log_filename + ".lock"
    log_content = f"[{now_str()}] {caller_name}: {log}\n"
    try:
        lock = FileLock(log_lock_filename, timeout=10)
        with lock:
            with open(log_filename, 'a', encoding='utf-8') as f:
                f.write(log_content)
    except Exception as e:
        print(f"로그 파일 쓰기 오류 발생: {e}")

OpLog("Program started.2025.11.27.001", bLines=True)

## 그래픽 출력 함수
def ShowPlt(plt):
    # plt.tight_layout()
    # plt.show(block = False)
    # plt.pause(3)
    plt.close()
   
## 메타 클래스 - 전역 설정 및 데이터 정보 관리
class MyMeta():
    def __init__(self):
        self._original_files, self._fuse_files, self._image_folder =   ViewDir()
        self._color_to_label = self.get_unique_colors()
        self._num_classes = len(self._color_to_label)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_epochs=50
        self._lr=0.005
        self._result_csv = f"{BASE_DIR}/result.csv"

    def original_files(self):
        return self._original_files
    def fuse_files(self):
        return self._fuse_files
    def image_folder(self):
        return self._image_folder
    def device(self):
        return self._device
    def num_classes(self):
        return self._num_classes
    def color_to_label(self):
        return self._color_to_label
    def lr(self):
        return self._lr
    def num_epochs(self):
        return self._num_epochs
    def result_csv(self):
        return self._result_csv
 
    def get_unique_colors(self, max_classes=11):
        # ▶ 마스크 이미지에서 고유 색상을 추출하여 클래스 ID에 매핑
        color_list = []
        for mask_file in self._fuse_files:
            mask_path = os.path.join(self._image_folder, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if mask is None:
                continue
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            unique_colors = np.unique(mask.reshape(-1, 3), axis=0)

            for color in unique_colors:
                color_tuple = tuple(color)
                if color_tuple not in color_list:
                    color_list.append(color_tuple)  # 고유한 색상 저장

                # 클래스 개수가 max_classes개가 되면 중단
                if len(color_list) >= max_classes:
                    # 색상 리스트를 dict로 변환 (색상 -> 클래스 ID)
                    return {color: idx for idx, color in enumerate(color_list)}
        # 모든 마스크를 순회 후 반환 (색상 -> 클래스 ID)
        return {color: idx for idx, color in enumerate(color_list)}
## 데이터셋 확인 및 로드
def ViewDir():
    # ▶ 이미지와 마스크 파일 경로 확인 및 매칭
    # 폴더 내 모든 파일 목록 가져오기
    image_folder = os.path.join(BASE_DIR, "images")
    file_list = os.listdir(image_folder)

    # 원본 이미지(.jpg)와 fuse 이미지 매칭
    original_files = sorted([f for f in file_list if f.endswith(".jpg")])
    fuse_files = sorted([f for f in file_list if "fuse" in f])

    # 이미지 로드 및 확인
    image_pairs = []
    for orig_file in original_files:
        # 동일한 프레임의 fuse 파일 찾기
        base_name = orig_file.replace(".jpg", "")
        fuse_file = next((f for f in fuse_files if base_name in f), None)

        if fuse_file:
            # 원본과 마스크 로드
            img_path = os.path.join(image_folder, orig_file)
            mask_path = os.path.join(image_folder, fuse_file)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            if img is not None and mask is not None:
                image_pairs.append((img, mask))

    # 최종적으로 로드된 이미지 쌍 개수 출력
    Lines(f"{len(image_pairs)} pairs")
    return original_files, fuse_files, image_folder

## 이미지 시각화 함수
def ViewImage(original_files,fuse_files,image_folder):
    sample_image_pairs = []
    num_samples = 5

    for orig_file in original_files[:num_samples]:
        base_name = orig_file.replace(".jpg", "")
        fuse_file = next((f for f in fuse_files if base_name in f), None)

        if fuse_file:
            sample_image_pairs.append((orig_file, fuse_file))

    # 이미지 5쌍 시각화
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

    for i, (orig_file, fuse_file) in enumerate(sample_image_pairs):
        # 이미지 로드
        orig_img = cv2.imread(os.path.join(image_folder, orig_file))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)  # OpenCV BGR → RGB 변환

        fuse_img = cv2.imread(os.path.join(image_folder, fuse_file))
        fuse_img = cv2.cvtColor(fuse_img, cv2.COLOR_BGR2RGB)  # 마스크도 RGB 변환

        # 시각화
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original: {orig_file}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(fuse_img)
        axes[i, 1].set_title(f"Fuse Mask: {fuse_file}")
        axes[i, 1].axis("off")
    ShowPlt(plt)

## 테스트용 디렉토리 및 이미지 뷰어
def TestDirView():
    original_files, fuse_files, image_folder =   ViewDir()
    ViewImage(original_files,fuse_files,image_folder)

TestDirView()

## meta 인스턴스 생성
MY_META = MyMeta()
## 고유 색상 및 클래스 매핑 출력
print(MY_META.get_unique_colors())

# ════════════════════════════════════════
# ▣ Transform/Dataset/DataLoader
# ════════════════════════════════════════

## 데이터 증강 및 전처리 함수
def GetTransfrom(type: str = "A", crop_size: int = 256) -> typing.Callable:
    # type (str): 사용할 변환 유형 ("A", "B", "C", "D").
    # crop_size (int): 이미지 출력 크기 (C, D 유형에서 사용).
    # return : Callable: PyTorch transforms.v2.Compose 객체.
    # ----------------------------------------
    # A 유형: 기본 증강 (Resize 기반)
    # ----------------------------------------
    if type == "A":
        ## ★ 데이터 증강을 포함한 학습용 변환
        train_transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_transform
        
    # ----------------------------------------
    # B 유형: 강력 증강 (기하학적/노이즈 추가, Resize 기반)
    # ----------------------------------------
    elif type == "B":
        train_transform = v2.Compose([
            v2.Resize((256, 256)),
            # 기존 증강
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            
            # 추가 증강
            v2.RandomRotation(degrees=15), 
            v2.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_transform

    # ----------------------------------------
    # C 유형: Segmentation 최적화 증강 (Perspective 추가, Sharpness 제외)
    # ----------------------------------------
    elif type == "C":
        train_transform = v2.Compose([
            # 1. 시점 변화 및 스케일 학습을 위해 RandomResizedCrop 사용 (Resize 대체)
            v2.RandomResizedCrop(crop_size, scale=(0.8, 1.2)), 
            
            # 2. 기하학적 증강
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15), 
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.RandomPerspective(distortion_scale=0.5, p=0.7), # ★ 핵심 추가
            
            # 3. 색상 및 노이즈 증강
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            
            # 4. 필수 전처리
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_transform
        
    # ----------------------------------------
    # D 유형: C 유형 + Sharpness 추가 (최강/최종 추천)
    # ----------------------------------------
    elif type == "D":
        train_transform = v2.Compose([
            # 1. 시점 변화 및 스케일 학습
            v2.RandomResizedCrop(crop_size, scale=(0.8, 1.2)), 
            
            # 2. 강력한 기하학적 증강
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15), 
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.RandomPerspective(distortion_scale=0.5, p=0.7),
            
            # 3. 색상, 노이즈 및 화질 증강
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            v2.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5), # ★ D 유형에 추가된 샤프니스 증강
            
            # 4. 필수 전처리
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_transform

    else:
        # A, B, C, D 유형 외의 잘못된 type이 입력되었을 때 예외 처리
        raise ValueError(f"Unknown transform type: {type}. Please use 'A', 'B', 'C', or 'D'.")
    

## 데이터셋 클래스 정의
class FootballDataset(Dataset):
    ## 축구 이미지 세그멘테이션 데이터셋 - RGB 마스크를 클래스 레이블로 변환
    def __init__(self, trainsform, image_files, mask_files, image_folder, color_to_label, train=False ):
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_folder = image_folder
        self.color_to_label = color_to_label  # 고정된 클래스 매핑
        self.train = train
        self.train_transform = trainsform 
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.image_folder, self.mask_files[idx])

        # 원본 이미지 로드
        img = Image.open(img_path).convert("RGB")

        # 마스크 로드 (RGB 모드)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # RGB 마스크를 클래스 레이블 맵으로 변환
        mask_class = np.zeros(mask.shape[:2], dtype=np.uint8)
        for color_tuple, label in self.color_to_label.items():
            color_array = np.array(color_tuple)
            mask_class[(mask == color_array).all(axis=-1)] = label

        mask = torch.tensor(mask_class, dtype=torch.long)

        # 데이터 증강 적용 (학습 시에만)
        if self.train:
            # 이미지와 마스크를 함께 변환
            img = T.ToTensor()(img) # 0-1 range
            img, mask = self.train_transform(img, mask)
        else:
            # 검증/테스트 시에는 리사이즈와 정규화만 적용
            transform = v2.Compose([
                v2.Resize((256, 256)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = transform(img)
            mask = torch.tensor(cv2.resize(mask_class, (256, 256), interpolation=cv2.INTER_NEAREST), dtype=torch.long)

        return img, mask

print("Define FootballDataset class")

## 데이터 로더 함수
def GetLoader(transform_type="A"):
    ## 학습/테스트 데이터 로더 생성 (8:2 분할)
    Lines("Load Dataset")

    color_to_label = MY_META.color_to_label()
    Lines(f"Unique colors found: {len(color_to_label)}")

    ## 데이터 로드
    original_files = MY_META.original_files()
    fuse_files = MY_META.fuse_files()
    image_folder = MY_META.image_folder()
    transform = GetTransfrom(transform_type)
    dataset = FootballDataset(transform, original_files, fuse_files, image_folder, color_to_label)
    
    ## 데이터셋 분할
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=test_size, random_state=42)

    ## 서브셋 생성
    train_dataset = torch.utils.data.Subset(FootballDataset(transform, original_files, fuse_files, image_folder, color_to_label, train=True), train_indices)
    test_dataset = torch.utils.data.Subset(FootballDataset(transform,original_files, fuse_files, image_folder, color_to_label, train=False), test_indices)

    ## DataLoader 생성
    Lines(f"Train Data: {len(train_dataset)}, Test Data: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    return train_loader, test_loader

## 데이터 로더 테스트 함수.
def TestLoader():
    train_loader, test_loader = GetLoader()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
TestLoader()

# ════════════════════════════════════════
# ▣ 평가 지표 및 모델 관리
# ════════════════════════════════════════
## 평가 지표 저장 함수.csv파일에 저장.
def save_metrics_to_csv(metrics, params_name, epochs, learnRate, epoch_index, data_set_name):
    mAP = metrics.get('mAP', 0.0)
    avg_loss = metrics.get('avg_loss', 0.0)
    ce_loss = metrics.get('ce_loss', 0.0)
    dice_loss = metrics.get('dice_loss', 0.0)
    accuracy = metrics.get('accuracy', 0.0)
    precision = metrics.get('precision', 0.0)
    recall = metrics.get('recall', 0.0)
    f1_score = metrics.get('f1_score', 0.0)
    
    new_data = {
        'Strategy': [params_name], 'Max_Epochs': [epochs], 'Epoch_Index': [epoch_index],
        'DataSet': [data_set_name], 'LearnRate': [learnRate], 
        'TotalLoss': [avg_loss], 'CELoss': [ce_loss], 'DiceLoss': [dice_loss],
        'Accuracy': [accuracy], 'mIoU': [mAP],
        'Precision': [precision], 'Recall': [recall], 'F1Score': [f1_score],
        'TN': [0], 'FP': [0], 'FN': [0], 'TP': [0]
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
    OpLog(f"Metrics saved: {data_set_name} Epoch {epoch_index}, mIoU: {mAP:.4f}")

## 평가 지표 계산 함수. 
def calculate_metrics(pred_masks, true_masks, num_classes):
    # ▶ mIoU, Accuracy, Precision, Recall, F1-Score 계산
    # Flatten
    pred_flat = pred_masks.flatten()
    true_flat = true_masks.flatten()
    
    # Pixel Accuracy
    correct = (pred_flat == true_flat).sum()
    total = len(true_flat)
    accuracy = correct / total
    
    # Per-class metrics
    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for cls in range(num_classes):
        pred_inds = (pred_flat == cls)
        target_inds = (true_flat == cls)
        
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        
        # IoU
        if union == 0:
            iou_list.append(float('nan')) # Ignore if class not present
        else:
            iou_list.append(intersection / union)
            
        # Precision & Recall
        pred_sum = pred_inds.sum()
        target_sum = target_inds.sum()
        
        prec = intersection / pred_sum if pred_sum > 0 else 0.0
        rec = intersection / target_sum if target_sum > 0 else 0.0
        
        precision_list.append(prec)
        recall_list.append(rec)
        
        # F1
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        
    # Mean metrics (ignoring NaNs for IoU)
    mIoU = np.nanmean(iou_list)
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)
    
    return {
        'accuracy': accuracy,
        'mAP': mIoU, # Using mAP field for mIoU as requested/common
        'precision': mean_precision,
        'recall': mean_recall,
        'f1_score': mean_f1
    }

## Dice Loss 정의
class DiceLoss(nn.Module):
    # ▶ 세그멘테이션용 Dice Loss
    def __init__(self, num_classes, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, outputs, targets):
        # ▶ outputs: (B,C,H,W) logits, targets: (B,H,W) class indices
        # Softmax를 적용하여 확률로 변환
        probs = F.softmax(outputs, dim=1)
        
        # One-hot encoding for targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets_one_hot.contiguous().view(-1)
        
        # Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice

## 평가 이미지 저장 함수 
def SaveEvalImages(images, true_masks, pred_masks, modelName, numEpoch, epochIndex):
    # ▶ 평가 결과 이미지 저장 (원본, GT, 예측)
    save_dir = f"{BASE_DIR}/modelfiles/eval_images"
    makedirs(save_dir)  # 디렉토리가 없으면 생성
    saveFileName = f"{save_dir}/{modelName}_{numEpoch}_{epochIndex}.png"

    # 시각화를 위해 0번째 배치 아이템 사용
    img = images[0].cpu().permute(1, 2, 0).numpy()
    true_mask = true_masks[0].cpu().numpy()
    pred_mask = pred_masks[0].cpu().numpy()

    # 역정규화
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Epoch {epochIndex} - Original")
    axes[0].axis('off')

    axes[1].imshow(true_mask, cmap='jet')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='jet')
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    plt.suptitle(f"Model: {modelName} - Epoch {epochIndex}/{numEpoch}", fontsize=16)
    plt.tight_layout()
    plt.savefig(saveFileName)
    plt.close(fig)
    OpLog(f"Saved evaluation image to {saveFileName}")

## 테스트용 평가 이미지 저장 함수
def TestSaveEvalImages():
    train_loader, test_loader = GetLoader()
    model = UNet(MY_META.num_classes())
    model.load_state_dict(torch.load(f"{BASE_DIR}/modelfiles/UNet_50_1.pth"))
    model.eval()
    SaveEvalImages(*next(iter(test_loader)), model.GetMyName(), 50, 1)
#TestSaveEvalImages()

# ════════════════════════════════════════
# ▣ 모델 클래스 정의
# ════════════════════════════════════════
## 모델 저장 함수 
def SaveModel(model,epochs,epochIndex):
    save_dir = f"{BASE_DIR}/modelfiles"
    makedirs(save_dir)
    torch.save(model.state_dict(), f"{save_dir}/{model.GetMyName()}_{epochs}_{epochIndex}.pth")

## 모델 로드 함수
def LoadModel(model,epochs,epochIndex):
    model.load_state_dict(torch.load(f"{BASE_DIR}/modelfiles/{model.GetMyName()}_{epochs}_{epochIndex}.pth"))
    model.eval()

## BaseModel 클래스
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self._name = "BaseModel"
        self._epochs = MY_META.num_epochs()
        self._lr = MY_META.lr()
        self._transform_type = "A"
        self._num_classes = MY_META.num_classes()

    ## 모델 이름 반환
    def getMyName(self):
        return self._name

    def getEpochs(self):
        return self._epochs
    def getLr(self):
        return self._lr
    def getTransformType(self):
        return self._transform_type
    
    ## 모델 학습 함수 CE Loss + Dice Loss 결합
    def fit(self, train_loader, test_loader, num_epochs, lr):
        device = MY_META.device()
        ce_criterion = nn.CrossEntropyLoss()
        dice_criterion = DiceLoss(num_classes=MY_META.num_classes())
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch in range(num_epochs):
            self.train()
            running_ce_loss = 0.0
            running_dice_loss = 0.0
            running_total_loss = 0.0
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                
                ### Cross Entropy Loss와 Dice Loss를 결합
                ce_loss = ce_criterion(outputs, masks)
                dice_loss = dice_criterion(outputs, masks)
                loss = ce_loss + dice_loss
                
                loss.backward()
                optimizer.step()
                
                running_ce_loss += ce_loss.item()
                running_dice_loss += dice_loss.item()
                running_total_loss += loss.item()

            train_ce_loss = running_ce_loss / len(train_loader)
            train_dice_loss = running_dice_loss / len(train_loader)
            train_total_loss = running_total_loss / len(train_loader)
            OpLog(f"Epoch {epoch+1} Train - CE Loss: {train_ce_loss:.4f}, Dice Loss: {train_dice_loss:.4f}, Total Loss: {train_total_loss:.4f}")

            ### 검증 단계
            self.eval()
            all_preds, all_masks = [], []
            eval_ce_loss = 0.0
            eval_dice_loss = 0.0
            eval_total_loss = 0.0
            
            with torch.no_grad():
                for images, masks in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Eval]"):
                    images, masks = images.to(device), masks.to(device)
                    outputs = self(images)
                    
                    #### Cross Entropy Loss와 Dice Loss 계산
                    ce_loss = ce_criterion(outputs, masks)
                    dice_loss = dice_criterion(outputs, masks)
                    total_loss = ce_loss + dice_loss
                    
                    eval_ce_loss += ce_loss.item()
                    eval_dice_loss += dice_loss.item()
                    eval_total_loss += total_loss.item()
                    
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds.cpu())
                    all_masks.append(masks.cpu())

            all_preds = torch.cat(all_preds)
            all_masks = torch.cat(all_masks)
            metrics = calculate_metrics(all_preds, all_masks, MY_META.num_classes())
            metrics['avg_loss'] = eval_total_loss / len(test_loader)
            metrics['ce_loss'] = eval_ce_loss / len(test_loader)
            metrics['dice_loss'] = eval_dice_loss / len(test_loader)
            
            OpLog(f"Epoch {epoch+1} Eval - CE Loss: {metrics['ce_loss']:.4f}, Dice Loss: {metrics['dice_loss']:.4f}, Total Loss: {metrics['avg_loss']:.4f}, mIoU: {metrics['mAP']:.4f}")
            save_metrics_to_csv(metrics, self.GetMyName(), num_epochs, lr, epoch + 1, "Test")
            SaveModel(self, num_epochs, epoch + 1)

            ### 에포크별 결과 시각화
            self.eval()
            with torch.no_grad():
                test_iter = iter(test_loader)
                num_samples_to_show = 3
                plt.figure(figsize=(15, 5 * num_samples_to_show))
                
                for i in range(num_samples_to_show):
                    try:
                        images, masks = next(test_iter)
                    except StopIteration:
                        break

                    images, masks = images.to(device), masks.to(device)
                    outputs = self(images)
                    preds = torch.argmax(outputs, dim=1)

                    ### 시각화를 위해 0번째 배치 아이템 사용
                    img = images[0].cpu().permute(1, 2, 0).numpy()
                    mask = masks[0].cpu().numpy()
                    pred = preds[0].cpu().numpy()

                    ### 역정규화
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)

                    plt.subplot(num_samples_to_show, 3, i * 3 + 1)
                    plt.imshow(img)
                    plt.title(f"Epoch {epoch+1} - Original")
                    plt.axis('off')

                    plt.subplot(num_samples_to_show, 3, i * 3 + 2)
                    plt.imshow(mask, cmap='jet')
                    plt.title("Ground Truth")
                    plt.axis('off')

                    plt.subplot(num_samples_to_show, 3, i * 3 + 3)
                    plt.imshow(pred, cmap='jet')
                    plt.title("Prediction")
                    plt.axis('off')

                plt.suptitle(f"Epoch {epoch+1} Results", fontsize=16)
                ShowPlt(plt)
                SaveEvalImages(images.cpu(), masks.cpu(), preds.cpu(), self.GetMyName(), num_epochs, epoch + 1)
            scheduler.step()

## U-Net 모델 정의.(기본 U-Net)
class UNet(BaseModel):
    ## 기본 U-Net 모델 (BatchNorm 포함)
    def __init__(self,transform_type="A", epochs=50, lr=0.005):
        super(UNet, self).__init__()
        self._name = "UNet"
        self._transform_type = transform_type
        self._epochs = epochs
        self._lr = lr
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        self.final_conv = nn.Conv2d(64,self._num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        return self.final_conv(d1)  # logits 출력

## 개선된 U-Net 모델 정의
class AdvancedUNet(BaseModel):
    ## Attention 메커니즘을 적용한 개선된 U-Net
    ## Attention Block 정의
    class AttentionBlock(nn.Module):
        # Attention Gate 블록 - 중요 영역에만 집중하도록 가중치 계산
        # 디코더가 인코더 특징을 받을 때, 배경은 억제하고 관심 객체(선수, 공 등)는 강조
        def __init__(self, F_g, F_l, F_int):
            ## F_g: 디코더에서 올라오는 특징 채널 수 (gating signal)
            ## F_l: 인코더에서 받는 특징 채널 수 (skip connection)
            ## F_int: 중간 레이어 채널 수 (attention 계산용)
            super(AdvancedUNet.AttentionBlock, self).__init__()
            ## W_g: 디코더 특징(g)을 중간 차원으로 변환
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
                nn.BatchNorm2d(F_int)
            )
            
            ## W_x: 인코더 특징(x)을 중간 차원으로 변환
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
                nn.BatchNorm2d(F_int)
            )
            
            ## psi: 두 특징을 결합하여 0~1 사이의 attention 가중치 생성
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), 
                nn.BatchNorm2d(1), 
                nn.Sigmoid()  # 0~1 범위로 정규화 (1=중요, 0=배경)
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            ## g: 디코더에서 올라온 특징 (어디를 봐야 하는지 힌트)
            ## x: 인코더에서 받은 특징 (실제 세부 정보)
            
            ## 1. 두 특징을 같은 차원으로 변환
            g1 = self.W_g(g)  # 디코더 특징 변환
            x1 = self.W_x(x)  # 인코더 특징 변환
            
            ## 2. 두 특징을 더하고 활성화 (어떤 영역이 중요한지 계산)
            psi = self.relu(g1 + x1)
            
            ## 3. Sigmoid로 0~1 사이의 가중치 맵 생성
            ## 중요한 영역은 1에 가깝고, 배경은 0에 가까움
            psi = self.psi(psi)
            
            ## 4. 인코더 특징에 가중치를 곱하여 중요한 부분만 강조
            ## 예: 선수/공 영역은 그대로, 배경은 억제
            return x * psi  # element-wise multiplication
    def __init__(self,transform_type="A", epochs=50, lr=0.005):
        super(AdvancedUNet, self).__init__()
        self._name = "AdvancedUNet"
        self._transform_type = transform_type
        self._epochs = epochs
        self._lr = lr
  
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            )

        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = self.AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = self.AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = self.AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, self._num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.upconv3(b)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat((e3_att, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat((e2_att, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat((e1_att, d1), dim=1)
        d1 = self.decoder1(d1)

        return self.final_conv(d1)

## 전이학습 U-Net 모델 정의
class TransferLearningUNet(BaseModel):
    ## ResNet34 백본을 사용한 전이학습 U-Net
    def __init__(self,transform_type="A", epochs=50, lr=0.005):
        super(TransferLearningUNet, self).__init__()
        self._name = "AdvancedUNet"
        self._transform_type = transform_type
        self._epochs = epochs
        self._lr = lr
        self._name = "TransferLearningUNet_ResNet34"
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.bottleneck = resnet.layer4
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True))
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.final_conv = nn.Conv2d(64, self._num_classes, kernel_size=1)

    ## 순전파 정의
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        b = self.bottleneck(e4)

        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        # Add final upsampling to restore original size
        out = nn.functional.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        return self.final_conv(out)

print("Defined UNet, AdvancedUNet, and TransferLearningUNet models")

## 학습 실행 함수
def TrainModel(model_type='UNet', transform_type="A", numEpochs=50, learningRate=0.005): 
    ## 모델 학습 실행 (파라미터 커스터마이징 가능)
    OpLog(f"TrainModel started for {model_type}", bLines=True)
    device = MY_META.device()
    train_loader, test_loader = GetLoader(transform_type=transform_type)

    ## 모델 인스턴스 생성
    if model_type == 'UNet':
        model = UNet(transform_type==transform_type, epochs=numEpochs, lr=learningRate).to(device)
    elif model_type == 'AdvancedUNet':
        model = AdvancedUNet(transform_type=transform_type, epochs=numEpochs, lr=learningRate).to(device)
    elif model_type == 'TransferLearningUNet':
        model = TransferLearningUNet(transform_type=transform_type, epochs=numEpochs, lr=learningRate).to(device)
    else:
        raise ValueError("Unknown model_type")
    ## 멤버 메서드 fit을 사용하여 학습 수행
    model.fit(train_loader=train_loader, test_loader=test_loader, num_epochs=numEpochs, lr=learningRate) # 에포크 및 학습률 조정
    ## 학습 후 모델과 테스트 로더, 디바이스 반환
    return model, test_loader

# ▶ 테스트 및 시각화
def TestModel(model, test_loader):
    ## 테스트 데이터로 모델 평가 및 결과 시각화
    device = MY_META.device()
    model.eval()
    
    # 디렉토리가 없으면 생성
    save_dir = f"{BASE_DIR}/modelfiles/test_images"
    makedirs(save_dir)
    save_filename = f"{save_dir}/{model.GetMyName()}_test_results.png"

    with torch.no_grad():
        test_iter = iter(test_loader)
        num_samples_to_show = 10
        fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(15, 5 * num_samples_to_show))

        for i in range(num_samples_to_show):
            try:
                images, masks = next(test_iter)
            except StopIteration:
                break

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 시각화를 위해 0번째 배치 아이템 사용
            img = images[0].cpu().permute(1, 2, 0).numpy()
            mask = masks[0].cpu().numpy()
            pred = preds[0].cpu().numpy()

            # 역정규화
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='jet')
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='jet')
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

        plt.suptitle(f"Test Results for {model.GetMyName()}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_filename)  # 이미지 파일로 저장
        OpLog(f"Saved test result image to {save_filename}")
        ShowPlt(plt)  # 화면에 표시

def Run():
    # UNet, AdvancedUNet, TransferLearningUNet 중 선택하여 학습
    model_types_to_train = ['UNet', 'AdvancedUNet', 'TransferLearningUNet']
    
    for model_type in model_types_to_train:
        model, test_loader = TrainModel(model_type=model_type)
        TestModel(model, test_loader)
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
#Run()

def Run_SingleModel(transform_type="A" , modelType='AdvancedUNet', numEpochs=20, learningRate=0.001):
    model, test_loader = TrainModel(transform_type=transform_type, model_type=modelType, numEpochs=numEpochs, learningRate=learningRate)
    TestModel(model, test_loader)
    # 메모리 정리
    del model
    torch.cuda.empty_cache()

# ════════════════════════════════════════
# ▣ 모델별 학습 실행 코드
# ════════════════════════════════════════

# UNet: 기본 U-Net 모델 (가장 빠르지만 성능은 중간)
# Run_SingleModel("A",'UNet', numEpochs=30, learningRate=0.001)
# Run_SingleModel("D",'UNet', numEpochs=30, learningRate=0.001)


# AdvancedUNet: Attention 메커니즘 적용 (중간 속도, 높은 성능)
# Run_SingleModel("A",'AdvancedUNet', numEpochs=30, learningRate=0.001)
# Run_SingleModel("D",'AdvancedUNet', numEpochs=30, learningRate=0.001)


# TransferLearningUNet: ResNet34 백본 사용 (느리지만 최고 성능)
Run_SingleModel("A",'TransferLearningUNet', numEpochs=30, learningRate=0.0005)
# Run_SingleModel("D",'TransferLearningUNet', numEpochs=30, learningRate=0.0005)


        
