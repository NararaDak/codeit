import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------------
# 1. 데이터셋 정의
# -----------------------------
class FootballSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        img = np.array(img)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask

# -----------------------------
# 2. U-Net 모델 정의
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.dconv_down1 = DoubleConv(3, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out

# -----------------------------
# 3. Loss 함수 정의 (CrossEntropy + Dice)
# -----------------------------
def dice_loss(pred, target, smooth=1.0):
    pred = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()

    intersection = torch.sum(pred * target_onehot)
    union = torch.sum(pred) + torch.sum(target_onehot)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        d_loss = dice_loss(pred, target)
        return ce_loss + d_loss

# -----------------------------
# 4. 평가 지표 정의 (Pixel Accuracy, mIoU)
# -----------------------------
def pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    correct = (pred_classes == target).float().sum()
    total = torch.numel(target)
    return correct / total

def mean_iou(pred, target, num_classes=11):
    pred_classes = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred_classes == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(torch.tensor(1.0))
        else:
            ious.append(intersection / union)
    return torch.mean(torch.stack(ious))

# -----------------------------
# 5. 학습 루프
# -----------------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, device="cuda"):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        val_acc = 0
        val_miou = 0
        model.eval()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_acc += pixel_accuracy(outputs, masks).item()
                val_miou += mean_iou(outputs, masks, num_classes=11).item()

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Pixel Acc: {val_acc/len(val_loader):.4f}, "
              f"mIoU: {val_miou/len(val_loader):.4f}")

# -----------------------------
# 6. 클래스별 색상 맵 시각화
# -----------------------------
CLASS_COLORS = {
    0: (0, 0, 0),         # Background
    1: (255, 255, 255),   # Ball
    2: (0, 0, 255),       # Player
    3: (255, 0, 0),       # Referee
    4: (255, 255, 0),     # Goalpost
    5: (0, 255, 255),     # Field line
    6: (128, 0, 128),     # Audience
    7: (255, 165, 0),     # Advertisement board
    8: (135, 206, 235),   # Sky
    9: (0, 128, 0),       # Grass
    10: (192, 192, 192),  # Other objects
}

def decode_segmap(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        color_mask[mask == cls_id] = color
    return color_mask

def visualize_prediction(image, mask, pred):
    image = image.permute(1,2,0).cpu().numpy()
    mask = mask.cpu().numpy()
    pred_classes = torch.argmax(pred, dim=0).cpu().numpy()

    mask_color = decode_segmap(mask)
    pred_color = decode_segmap(pred_classes)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[1].imshow(mask_color)
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred_color)
    axes[2].set_title("Predicted Mask")
    for ax in axes:
        ax.axis("off")
    plt.show()

# -----------------------------
# 7. 실행 예시
# -----------------------------
#image_dir = "path/to/images"