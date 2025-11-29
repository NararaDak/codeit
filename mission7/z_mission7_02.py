#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_pet_transfer.py
#
# Oxford-IIIT Pet Dataset용 통합 학습 스크립트
# - SSD (detection) : torchvision SSD300 VGG16 사용
# - Classification   : EfficientNet-B3 / Swin-Tiny / MobileNetV2 / MobileNetV3 (전이학습)
#
# 사용법 예:
# python train_pet_transfer.py --base_dir /path/to/pet_data --model EfficientNetB3 --epochs 10 --batch_size 32
#
# 주의: torchvision, torch, PIL 등 기본 라이브러리 설치 필요. timm은 사용하지 않았습니다.
#
# (작성자 노트) 더 많은 한국어 AI 자료: https://gptonline.ai/ko/
# ------------------------------------------------------------------------------

import os
import sys
import time
import argparse
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T

# Detection models
from torchvision.models.detection import ssd300_vgg16
try:
    from torchvision.models.detection.ssd import SSD300_VGG16_Weights
    SSD_WEIGHTS_ENUM_AVAILABLE = True
except Exception:
    SSD_WEIGHTS_ENUM_AVAILABLE = False

# -------------------------
# 유틸리티
# -------------------------
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def makedirs(d):
    os.makedirs(d, exist_ok=True)

def save_checkpoint(state, path):
    makedirs(os.path.dirname(path))
    torch.save(state, path)

# -------------------------
# 데이터 경로 및 기본 읽기
# -------------------------
def read_dataset_files(base_dir):
    """
    base_dir 구조 가정 (Kaggle Oxford-IIIT Pet):
    base_dir/
      images/images/...jpg
      annotations/annotations/xmls/...xml
      annotations/annotations/trainval.txt
      annotations/annotations/test.txt
    """
    base_dir = str(base_dir)
    trainval_file_path = os.path.join(base_dir, "annotations", "annotations", "trainval.txt")
    test_file_path = os.path.join(base_dir, "annotations", "annotations", "test.txt")
    image_dir = os.path.join(base_dir, "images", "images")
    xml_dir = os.path.join(base_dir, "annotations", "annotations", "xmls")

    df_trainval = pd.read_csv(trainval_file_path, sep="\s+", header=None)
    df_trainval.columns = ["Image", "ClassID", "Species", "BreedID"]

    df_test = pd.read_csv(test_file_path, sep="\s+", header=None)
    df_test.columns = ["Image", "ClassID", "Species", "BreedID"]

    return df_trainval, df_test, image_dir, xml_dir

# -------------------------
# Dataset: Detection (VOCDataset)
# -------------------------
class VOCDataset(Dataset):
    """
    Detection dataset: returns (image_tensor, target_dict)
    target_dict: {'boxes': Tensor[N,4], 'labels': Tensor[N]}
    Expects xml annotations in PASCAL VOC format (filename, object/bndbox/name)
    """
    def __init__(self, image_dir, annotation_dir, image_list, classes, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        # image_list may be list of base names without extension
        self.image_list = [os.path.splitext(x)[0] for x in image_list]
        self.classes = classes  # e.g. ["background","dog","cat"] or mapping of breed names (not used here)
        self.transform = transform
        # filter only those with xml
        filtered = []
        for im in self.image_list:
            xml_path = os.path.join(self.annotation_dir, im + ".xml")
            if os.path.exists(xml_path):
                filtered.append(im)
        self.image_list = filtered

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        base = self.image_list[idx]
        img_path = os.path.join(self.image_dir, base + ".jpg")
        xml_path = os.path.join(self.annotation_dir, base + ".xml")

        img = Image.open(img_path).convert("RGB")
        # parse xml
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            # 기본적으로 breed name들이 들어있음 -> 여기서는 species 구분 혹은 dog/cat
            # If classes contain breed names, map accordingly. For now we map dog/cat using df (but VOC xml may have breed)
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            # label index - the VOCDataset will expect label indices; if classes are ["background","dog","cat"], try to map by word
            # Simple heuristic: if name contains 'dog' or 'cat' else 1
            lab = 1  # default dog
            if "cat" in name.lower():
                lab = 2
            labels.append(lab)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        if self.transform:
            # transform should accept PIL image and return tensor
            img = self.transform(img)

        target = {"boxes": boxes, "labels": labels}
        return img, target

# -------------------------
# Dataset: Classification
# -------------------------
class PetClassificationDataset(Dataset):
    """
    Classification dataset for breed/species classification using df (trainval/test)
    Returns: image_tensor, label (int)
    Here label choice can be BreedID (0..36) or Species(0/1) depending on use-case.
    """
    def __init__(self, image_dir, df, target_col="BreedID", transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_col = target_col
        self.images = self.df["Image"].values
        self.targets = self.df[self.target_col].values
        # convert to 0-based labels
        self.targets = self.targets.astype(int) - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        base = str(self.images[idx])
        img_path = os.path.join(self.image_dir, base + ".jpg")
        img = Image.open(img_path).convert("RGB")
        y = int(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        return img, y

# -------------------------
# Transforms (model-type별)
# -------------------------
def get_transforms(model_type, is_train=True):
    """
    모델 타입에 따라 적절한 transform을 제공.
    - SSD: ToTensor (pil->[0,1]) + Resize(300)
    - EfficientNetB3: 이미지 크기 300 (EfficientNet-B3 권장 300x300)
    - SwinTiny: 224x224
    - MobileNetV2/V3: 224x224
    """
    if model_type == "SSD":
        if is_train:
            return T.Compose([
                T.Resize((300,300)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # converts to [0,1]
            ])
        else:
            return T.Compose([
                T.Resize((300,300)),
                T.ToTensor(),
            ])

    elif model_type == "EfficientNetB3":
        if is_train:
            return T.Compose([
                T.RandomResizedCrop(300),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            return T.Compose([
                T.Resize(320),
                T.CenterCrop(300),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    elif model_type == "SwinTiny":
        if is_train:
            return T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    elif model_type in ("MobileNetV2","MobileNetV3"):
        if is_train:
            return T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
    else:
        raise ValueError(f"Unknown model_type for transforms: {model_type}")

# -------------------------
# 모델 클래스 (BasicModel 상속)
# -------------------------
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()

    def GetMyName(self):
        return "BasicModel"

    def setup_training(self):
        raise NotImplementedError

# SSDTransfer: 기존 코드와 유사
class SSDTransfer(BasicModel):
    def __init__(self, num_classes, gubun="freeze", learn_rate=1e-3, backbone_lr_ratio=0.1, device=torch.device("cpu")):
        super().__init__()
        self.num_classes = num_classes
        self.gubun = gubun
        self.learn_rate = learn_rate
        self.backbone_lr_ratio = backbone_lr_ratio
        self.device = device

        # SSD weights enum 사용 가능하면 그걸 사용, 아니면 pretrained=True 시도
        if SSD_WEIGHTS_ENUM_AVAILABLE:
            weights = SSD300_VGG16_Weights.DEFAULT
            self.model = ssd300_vgg16(weights=weights)
        else:
            # fallback
            try:
                self.model = ssd300_vgg16(pretrained=True)
            except Exception:
                self.model = ssd300_vgg16(weights=None)
        # set num_classes properly (classification head)
        # torchvision SSD implementation: set model.head.classification_head.num_classes
        try:
            self.model.head.classification_head.num_classes = num_classes
        except Exception:
            # older/newer implementations may differ; attempt to replace head if necessary
            pass

    def GetMyName(self):
        return "SSD300"

    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

    def setup_training(self):
        # freeze or not
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if "head" in name or "classification_head" in name or "regression_head" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                if self.gubun == "freeze":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    backbone_params.append(param)

        params_group = [{'params': head_params, 'lr': self.learn_rate}]
        if self.gubun != "freeze" and backbone_params:
            eff_lr = self.learn_rate * self.backbone_lr_ratio
            params_group.append({'params': backbone_params, 'lr': eff_lr})

        optimizer = torch.optim.SGD(params_group, lr=self.learn_rate, momentum=0.9, weight_decay=5e-4)
        return optimizer

# Helper to safely get weights enum or fallback
def _get_weights_enum(model_name):
    try:
        # e.g. models.EfficientNet_B3_Weights.IMAGENET1K_V1
        weights_enum = getattr(torchvision.models, f"{model_name}_Weights")
        # pick the most likely enum name
        enum_vals = list(weights_enum)
        return enum_vals[0]
    except Exception:
        return None

# EfficientNet-B3 Transfer
class EfficientNetB3Transfer(BasicModel):
    def __init__(self, num_classes, gubun="freeze", learn_rate=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        self.gubun = gubun
        self.learn_rate = learn_rate
        self.backbone_lr_ratio = backbone_lr_ratio

        # torchvision EfficientNet-B3
        try:
            weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
            self.model = torchvision.models.efficientnet_b3(weights=weights)
        except Exception:
            # fallback: pretrained=True legacy
            self.model = torchvision.models.efficientnet_b3(pretrained=True)

        # classifier typically model.classifier[1]
        try:
            in_f = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_f, num_classes)
        except Exception:
            # alternative structure
            self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.model.classifier.in_features, num_classes))

    def GetMyName(self):
        return "EfficientNetB3"

    def forward(self, x):
        return self.model(x)

    def setup_training(self):
        head_params = []
        backbone_params = []
        for name, param in self.model.named_parameters():
            if "classifier" in name or "fc" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                if self.gubun == "freeze":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    backbone_params.append(param)

        param_groups = [{'params': head_params, 'lr': self.learn_rate}]
        if self.gubun != "freeze" and backbone_params:
            param_groups.append({'params': backbone_params, 'lr': self.learn_rate * self.backbone_lr_ratio})

        return torch.optim.Adam(param_groups, lr=self.learn_rate)

# Swin Tiny Transfer
class SwinTinyTransfer(BasicModel):
    def __init__(self, num_classes, gubun="freeze", learn_rate=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        self.gubun = gubun
        self.learn_rate = learn_rate
        self.backbone_lr_ratio = backbone_lr_ratio

        # torchvision swin_t
        try:
            weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_t(weights=weights)
        except Exception:
            # fallback
            self.model = torchvision.models.swin_t(pretrained=True)

        # replace head
        try:
            in_f = self.model.head.in_features
            self.model.head = nn.Linear(in_f, num_classes)
        except Exception:
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def GetMyName(self):
        return "SwinTiny"

    def forward(self, x):
        return self.model(x)

    def setup_training(self):
        head_params = []
        backbone_params = []
        for name, param in self.model.named_parameters():
            if "head" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                if self.gubun == "freeze":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    backbone_params.append(param)

        param_groups = [{'params': head_params, 'lr': self.learn_rate}]
        if self.gubun != "freeze" and backbone_params:
            param_groups.append({'params': backbone_params, 'lr': self.learn_rate * self.backbone_lr_ratio})
        return torch.optim.Adam(param_groups, lr=self.learn_rate)

# MobileNetV2 Transfer
class MobileNetV2Transfer(BasicModel):
    def __init__(self, num_classes, gubun="freeze", learn_rate=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        self.gubun = gubun
        self.learn_rate = learn_rate
        self.backbone_lr_ratio = backbone_lr_ratio
        try:
            weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
            self.model = torchvision.models.mobilenet_v2(weights=weights)
        except Exception:
            self.model = torchvision.models.mobilenet_v2(pretrained=True)

        # classifier replace
        try:
            in_f = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_f, num_classes)
        except Exception:
            self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.model.classifier.in_features, num_classes))

    def GetMyName(self):
        return "MobileNetV2"

    def forward(self, x):
        return self.model(x)

    def setup_training(self):
        head_params = []
        backbone_params = []
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                if self.gubun == "freeze":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    backbone_params.append(param)
        param_groups = [{'params': head_params, 'lr': self.learn_rate}]
        if self.gubun != "freeze" and backbone_params:
            param_groups.append({'params': backbone_params, 'lr': self.learn_rate * self.backbone_lr_ratio})
        return torch.optim.Adam(param_groups, lr=self.learn_rate)

# MobileNetV3 Large Transfer
class MobileNetV3Transfer(BasicModel):
    def __init__(self, num_classes, gubun="freeze", learn_rate=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        self.gubun = gubun
        self.learn_rate = learn_rate
        self.backbone_lr_ratio = backbone_lr_ratio
        try:
            weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.model = torchvision.models.mobilenet_v3_large(weights=weights)
        except Exception:
            self.model = torchvision.models.mobilenet_v3_large(pretrained=True)

        # classifier for mobilenet_v3_large: classifier[3]
        try:
            in_f = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_f, num_classes)
        except Exception:
            # fallback
            if hasattr(self.model, "classifier"):
                last = list(self.model.classifier.children())[-1]
                in_f = last.in_features
                self.model.classifier[-1] = nn.Linear(in_f, num_classes)

    def GetMyName(self):
        return "MobileNetV3"

    def forward(self, x):
        return self.model(x)

    def setup_training(self):
        head_params = []
        backbone_params = []
        for name, param in self.model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                if self.gubun == "freeze":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    backbone_params.append(param)
        param_groups = [{'params': head_params, 'lr': self.learn_rate}]
        if self.gubun != "freeze" and backbone_params:
            param_groups.append({'params': backbone_params, 'lr': self.learn_rate * self.backbone_lr_ratio})
        return torch.optim.Adam(param_groups, lr=self.learn_rate)

# -------------------------
# 모델 팩토리
# -------------------------
def MakeModel(model_type, num_classes, gubun, lr, ratio=0.1, device=torch.device("cpu")):
    if model_type == "SSD":
        model = SSDTransfer(num_classes=num_classes, gubun=gubun, learn_rate=lr, backbone_lr_ratio=ratio, device=device)
    elif model_type == "EfficientNetB3":
        model = EfficientNetB3Transfer(num_classes=num_classes, gubun=gubun, learn_rate=lr, backbone_lr_ratio=ratio)
    elif model_type == "SwinTiny":
        model = SwinTinyTransfer(num_classes=num_classes, gubun=gubun, learn_rate=lr, backbone_lr_ratio=ratio)
    elif model_type == "MobileNetV2":
        model = MobileNetV2Transfer(num_classes=num_classes, gubun=gubun, learn_rate=lr, backbone_lr_ratio=ratio)
    elif model_type == "MobileNetV3":
        model = MobileNetV3Transfer(num_classes=num_classes, gubun=gubun, learn_rate=lr, backbone_lr_ratio=ratio)
    else:
        raise ValueError("Unknown model_type: " + str(model_type))
    optimizer = model.setup_training()
    return model, optimizer

# -------------------------
# Trainer (Detection vs Classification 분기)
# -------------------------
class Trainer:
    def __init__(self, model, optimizer, device, model_type, train_loader, val_loader, test_loader, classes_map, checkpoint_dir, config):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.classes_map = classes_map
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        makedirs(self.checkpoint_dir)

    # classification train step
    def train_epoch_classification(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        loop = tqdm(self.train_loader, desc=f"Train cls Epoch {epoch}", leave=False)
        for imgs, labels in loop:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=correct/total)
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # classification evaluate
    def eval_classification(self, loader, desc="Val"):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            loop = tqdm(loader, desc=f"Eval {desc}", leave=False)
            for imgs, labels in loop:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return running_loss / total if total>0 else 0.0, correct / total if total>0 else 0.0

    # detection train epoch
    def train_epoch_detection(self, epoch):
        self.model.train()
        running_loss = 0.0
        loop = tqdm(self.train_loader, desc=f"Train det Epoch {epoch}", leave=False)
        for images, targets in loop:
            # images: tuple of tensors, targets: tuple of dicts
            images = [img.to(self.device) for img in images]
            t_targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            self.optimizer.zero_grad()
            loss_dict = self.model(images, t_targets)
            losses = sum(v for v in loss_dict.values())
            losses.backward()
            self.optimizer.step()
            running_loss += losses.item()
            loop.set_postfix(loss=losses.item())
        return running_loss / len(self.train_loader)

    # detection eval (compute mAP simplified)
    def eval_detection(self, loader):
        # for brevity take model outputs and compute a very simple metric: average score for predicted boxes
        # For proper mAP one should implement full matching — original code included more complex map calculation.
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            loop = tqdm(loader, desc="Eval det", leave=False)
            for images, targets in loop:
                images = [img.to(self.device) for img in images]
                preds = self.model(images)
                for p in preds:
                    if 'scores' in p and len(p['scores'])>0:
                        all_scores.append(p['scores'].cpu().numpy().mean())
        avg_score = float(np.mean(all_scores)) if len(all_scores)>0 else 0.0
        return {"avg_pred_score": avg_score}

    def fit(self, num_epochs):
        best_val = -1
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        for epoch in range(1, num_epochs+1):
            t0 = time.time()
            if self.model_type == "SSD":
                train_loss = self.train_epoch_detection(epoch)
                val_metrics = self.eval_detection(self.val_loader)
                # save checkpoint
                ckpt = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": self.config
                }
                save_path = os.path.join(self.checkpoint_dir, f"{self.model_type}_epoch_{epoch}.pth")
                save_checkpoint(ckpt, save_path)
                print(f"[{now_str()}] Epoch {epoch} (Det) train_loss={train_loss:.4f} val_avg_pred_score={val_metrics['avg_pred_score']:.4f} time={(time.time()-t0):.1f}s")
            else:
                train_loss, train_acc = self.train_epoch_classification(epoch)
                val_loss, val_acc = self.eval_classification(self.val_loader, desc="Validation")
                ckpt = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": self.config
                }
                save_path = os.path.join(self.checkpoint_dir, f"{self.model_type}_epoch_{epoch}.pth")
                save_checkpoint(ckpt, save_path)
                print(f"[{now_str()}] Epoch {epoch} (Cls) train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={(time.time()-t0):.1f}s")
            scheduler.step()

        # 마지막 테스트 평가
        print("== Final evaluation on test set ==")
        if self.model_type == "SSD":
            final_metrics = self.eval_detection(self.test_loader)
            print("Final test metrics (detection simplified):", final_metrics)
        else:
            test_loss, test_acc = self.eval_classification(self.test_loader, desc="Test")
            print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# -------------------------
# 실행 도우미: 데이터로더 구성 및 실행
# -------------------------
def build_dataloaders_for_detection(train_list, valid_list, test_list, image_dir, xml_dir, batch_size, classes_map, model_type):
    train_ds = VOCDataset(image_dir, xml_dir, train_list, classes_map, transform=get_transforms("SSD", is_train=True))
    val_ds = VOCDataset(image_dir, xml_dir, valid_list, classes_map, transform=get_transforms("SSD", is_train=False))
    test_ds = VOCDataset(image_dir, xml_dir, test_list, classes_map, transform=get_transforms("SSD", is_train=False))
    # For detection, collate_fn is necessary
    collate_fn = lambda batch: tuple(zip(*batch))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

def build_dataloaders_for_classification(df_trainval, df_test, image_dir, batch_size, valid_ratio=0.2, target_col="BreedID", model_type="EfficientNetB3"):
    # split trainval
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df_trainval, test_size=valid_ratio, random_state=42, stratify=df_trainval["BreedID"])
    train_transform = get_transforms(model_type, is_train=True)
    val_transform = get_transforms(model_type, is_train=False)
    test_transform = get_transforms(model_type, is_train=False)
    train_ds = PetClassificationDataset(image_dir, train_df, target_col=target_col, transform=train_transform)
    val_ds = PetClassificationDataset(image_dir, val_df, target_col=target_col, transform=val_transform)
    test_ds = PetClassificationDataset(image_dir, df_test, target_col=target_col, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, len(np.unique(train_ds.targets))

# -------------------------
# 메인 실행
# -------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df_trainval, df_test, image_dir, xml_dir = read_dataset_files(args.base_dir)

    # lists from files (these are basenames in trainval/test txt)
    trainval_list = df_trainval['Image'].tolist()
    test_list = df_test['Image'].tolist()

    # If user chooses detection (SSD), we will use xmls and VOCDataset.
    if args.model == "SSD":
        # split trainval into train/val
        from sklearn.model_selection import train_test_split
        train_list, valid_list = train_test_split(trainval_list, test_size=0.2, random_state=42)
        classes_map = ["background","dog","cat"]
        train_loader, val_loader, test_loader = build_dataloaders_for_detection(train_list, valid_list, test_list, image_dir, xml_dir, args.batch_size, classes_map, args.model)
        num_classes = len(classes_map)  # SSD needs num_classes (including background)
    else:
        # classification: default target is BreedID (37 classes). If user wants Species (dog/cat) set target_col="Species"
        target_col = "BreedID" if args.target == "breed" else "Species"
        train_loader, val_loader, test_loader, num_unique = build_dataloaders_for_classification(df_trainval, df_test, image_dir, args.batch_size, valid_ratio=0.2, target_col=target_col, model_type=args.model)
        # if BreedID used, there are 37 breeds -> labels 0..36
        num_classes = num_unique if target_col == "BreedID" else 2

    print(f"Num classes: {num_classes}")

    # Model building
    model, optimizer = MakeModel(args.model, num_classes=num_classes, gubun=args.gubun, lr=args.lr, ratio=args.ratio, device=device)
    # move classification model to device
    model = model.to(device)

    # Trainer
    checkpoint_dir = os.path.join(args.base_dir, "modelfiles", f"checkpoints_{args.model}_{args.gubun}_{int(time.time())}")
    cfg = {"model":args.model, "gubun":args.gubun, "lr":args.lr, "ratio":args.ratio}
    trainer = Trainer(model=model, optimizer=optimizer, device=device, model_type=args.model,
                      train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                      classes_map=None, checkpoint_dir=checkpoint_dir, config=cfg)

    # Fit
    trainer.fit(args.epochs)

# -------------------------
# Argparse
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pets Transfer Learning (SSD / EfficientNetB3 / SwinTiny / MobileNetV2/V3)")
    parser.add_argument("--base_dir", type=str, required=True, help="Path to pet_data root (images/, annotations/...)")
    parser.add_argument("--model", type=str, default="EfficientNetB3", choices=["SSD","EfficientNetB3","SwinTiny","MobileNetV2","MobileNetV3"], help="Model type")
    parser.add_argument("--gubun", type=str, default="freeze", choices=["freeze","partial","full"], help="freeze: only head, partial: head+backbone low lr, full: all trainable")
    parser.add_argument("--target", type=str, default="breed", choices=["breed","species"], help="Classification target: breed (37 classes) or species (2 classes)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ratio", type=float, default=0.1, help="backbone lr ratio when partial")
    args = parser.parse_args()
    print("Starting:", args)
    main(args)

# 끝으로 — 더 많은 한국어 AI 자료는 https://gptonline.ai/ko/ 에서 확인하세요 😊
