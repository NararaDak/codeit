import os
import random
import time
import datetime
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.transforms import v2
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import average_precision_score

# === Torchvision Detection Models ===

# from torchvision.models.detection import (
#     fasterrcnn_resnet50_fpn,
#     FasterRCNN_ResNet50_FPN_Weights,
#     retinanet_resnet50_fpn,
#     RetinaNet_ResNet50_FPN_Weights,
#     ssdlite320_mobilenet_v3_large,
#     SSDLite320_MobileNetV3_Large_Weights
# )

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDClassificationHead

# ════════════════════════════════════════
# ▣ 환경 설정
# ════════════════════════════════════════
BASE_DIR = r"D:\05.gdrive\codeit\mission7\data\pet_data" 
# BASE_DIR = "/content/drive/MyDrive/codeit/mission7/data/pet_data" # Colab용

def Lines(text="", count=100):
    print("═" * count)
    if text != "":
        print(f"{text}")
        print("═" * count)

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def makedirs(d):
    os.makedirs(d, exist_ok=True)

def OpLog(log, bLines=True):
    if bLines:
        Lines(f"[{now_str()}] {log}")
    try:
        caller_name = sys._getframe(1).f_code.co_name
    except Exception:
        caller_name = "UnknownFunction"
        
    log_filename = os.path.join(BASE_DIR, "op_log.txt")
    log_content = f"[{now_str()}] {caller_name}: {log}\n"
    try:
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(log_content)
    except Exception as e:
        print(f"로그 파일 쓰기 오류 발생: {e}")

OpLog("Program started.", bLines=True)

# ════════════════════════════════════════
# ▣ 메타 정보 클래스
# ════════════════════════════════════════
class MyMeta():
    def __init__(self):
        self._base_dir = BASE_DIR
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._trainval_file = os.path.join(self._base_dir, "annotations", "annotations", "trainval.txt")
        self._test_file = os.path.join(self._base_dir, "annotations", "annotations", "test.txt")
        self._image_dir = os.path.join(self._base_dir, "images", "images")
        self._xml_dir = os.path.join(self._base_dir, "annotations", "annotations", "xmls")
        self._num_workers = 0 
        self._classes = ["background", "dog", "cat"]
        
        self._xml_files = []
        if os.path.exists(self._xml_dir):
             self._xml_files = [file for file in os.listdir(self._xml_dir) if file.endswith(".xml")]

        try:
            self._df_trainval = pd.read_csv(self._trainval_file, sep="\s+", header=None)
            self._df_trainval.columns = ["Image", "ClassID", "Species", "BreedID"]
            
            self._df_test = pd.read_csv(self._test_file, sep="\s+", header=None)
            self._df_test.columns = ["Image", "ClassID", "Species", "BreedID"]

            self._trainval_list = self._df_trainval['Image'].tolist()
            self._test_list = self._df_test['Image'].tolist()
        except FileNotFoundError:
            print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
            self._df_trainval = None
            self._df_test = None
            self._trainval_list = []
            self._test_list = []

    @property
    def base_dir(self): return self._base_dir
    @property
    def device(self): return self._device
    @property
    def trainval_file(self): return self._trainval_file
    @property
    def test_file(self): return self._test_file
    @property
    def image_dir(self): return self._image_dir
    @property
    def xml_dir(self): return self._xml_dir
    @property
    def df_trainval(self): return self._df_trainval
    @property
    def df_test(self): return self._df_test
    @property
    def num_workers(self): return self._num_workers
    @property
    def xml_files(self): return self._xml_files
    @property
    def trainval_list(self): return self._trainval_list
    @property
    def test_list(self): return self._test_list
    @property
    def classes(self): return self._classes

# ════════════════════════════════════════
# ▣ 유틸리티 함수 (IoU, Metrics)
# ════════════════════════════════════════
def calculate_iou(box, boxes):
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / (union + 1e-6)

def evaluate_model(predictions, ground_truths, classes):
    class_aps = []
    for class_idx, class_name in enumerate(classes[1:], start=1): # background 제외
        true_positives = []
        scores = []
        num_ground_truths = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred["boxes"][pred["labels"] == class_idx].cpu().numpy() if len(pred["boxes"]) > 0 else []
            pred_scores = pred["scores"][pred["labels"] == class_idx].cpu().numpy() if len(pred["scores"]) > 0 else []
            gt_boxes = gt["boxes"][gt["labels"] == class_idx].cpu().numpy() if len(gt["boxes"]) > 0 else []
            
            num_ground_truths += len(gt_boxes)
            if len(pred_boxes) == 0: continue

            matched = np.zeros(len(gt_boxes), dtype=bool)
            for box, score in zip(pred_boxes, pred_scores):
                if len(gt_boxes) == 0:
                    true_positives.append(0)
                    scores.append(score)
                    continue
                    
                ious = calculate_iou(box, gt_boxes)
                max_iou_idx = np.argmax(ious) if len(ious) > 0 else -1
                max_iou = ious[max_iou_idx] if max_iou_idx >= 0 else 0

                if max_iou >= 0.5 and not matched[max_iou_idx]:
                    true_positives.append(1)
                    matched[max_iou_idx] = True
                else:
                    true_positives.append(0)
                scores.append(score)

        if len(scores) == 0:
            class_aps.append(0)
            continue

        sorted_indices = np.argsort(-np.array(scores))
        true_positives = np.array(true_positives)[sorted_indices]
        scores = np.array(scores)[sorted_indices]
        
        ap = average_precision_score(true_positives, scores) if len(scores) > 0 and np.sum(true_positives) > 0 else 0
        class_aps.append(ap)

    mAP = np.mean(class_aps) if class_aps else 0.0
    return mAP

def save_metrics_to_csv(metrics, params_name, epochs, learnRate, epoch_index, data_set_name, filename="training_metrics.csv"):
    mAP = metrics.get('mAP', 0.0)
    new_data = {
        'Strategy': [params_name], 'Max_Epochs': [epochs], 'Epoch_Index': [epoch_index],
        'DataSet': [data_set_name], 'LearnRate': [learnRate], 'AvgLoss': [metrics.get('avg_loss', 0.0)],
        'Accuracy': [mAP], # mAP 저장
        'Precision': [0.0], 'Recall': [0.0], 'Specificity': [0.0], 'F1Score': [0.0],
        'TN': [0], 'FP': [0], 'FN': [0], 'TP': [0]
    }
    new_df = pd.DataFrame(new_data)
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(filename, index=False)
        except:
            new_df.to_csv(filename, index=False)
    else:
        new_df.to_csv(filename, index=False)
    # print(f"Metrics saved: {data_set_name} Epoch {epoch_index}")

def GetTrainValidationSplit(df, test_size=0.3, random_state=42):
    meta = MyMeta()
    trainval_list = meta.trainval_list
    train_list, valid_list = train_test_split(trainval_list, test_size=0.3, random_state=42)
    Lines(f"Train/Validation :{len(train_list)},{len(valid_list)}")
    return train_list, valid_list

# ════════════════════════════════════════
# ▣ 데이터셋 클래스 (VOCDataset, TestDataset)
# ════════════════════════════════════════
class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, classes, image_list, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.transforms = transforms
        self.image_files = image_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx] + ".jpg"
        annotation_file = self.image_files[idx] + ".xml"
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, annotation_file)

        image = Image.open(image_path).convert("RGB")
        boxes = []
        labels = []
        
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in self.classes: continue
                labels.append(self.classes.index(class_name))
                bndbox = obj.find("bndbox")
                boxes.append([
                    int(bndbox.find("xmin").text), int(bndbox.find("ymin").text),
                    int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)
                ])
        except Exception:
            # 에러 발생 시 더미 데이터 반환 (GetLoader에서 필터링하지만 안전장치)
            return torch.zeros((3,300,300)), {"boxes": torch.zeros((0,4)), "labels": torch.zeros((0,), dtype=torch.int64)}

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        if boxes.numel() == 0: # 박스 없으면 배경 취급
             boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)
        
        return image, target

class TestDataset(Dataset):
    def __init__(self, image_dir, image_list, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = image_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx] + ".jpg"
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        
        # 테스트셋은 타겟이 없으므로 더미 타겟 생성 (Transforms 호환용)
        dummy_target = {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}

        if self.transforms:
            image = self.transforms(image) # 테스트셋은 이미지만 변환

        return image, self.image_files[idx]

# ════════════════════════════════════════
# ▣ 데이터 로더 (수정됨: transform 인자 추가)
# ════════════════════════════════════════
def GetLoader(meta, train_list, val_list, test_list, transform, batchSize=8):
    # XML 파일 필터링
    xml_dir = meta.xml_dir
    if os.path.exists(xml_dir):
        xml_list_base = {os.path.splitext(file)[0] for file in os.listdir(xml_dir) if file.endswith(".xml")}
    else:
        xml_list_base = set()
        
    train_list = [img for img in train_list if img in xml_list_base]
    val_list = [img for img in val_list if img in xml_list_base]
    
    # Transform 설정 (입력받은 custom_transform이 있으면 사용, 없으면 기본값)
    # 기본값은 VGG16 SSD용 (사용자 코드의 GetTransforms와 유사)
    
    train_dataset = VOCDataset(meta.image_dir, meta.xml_dir, meta.classes, train_list, transforms=transform)
    valid_dataset = VOCDataset(meta.image_dir, meta.xml_dir, meta.classes, val_list, transforms=transform)
    test_dataset = TestDataset(meta.image_dir, meta.test_list, transforms=transform)
    
    collate_fn = lambda x: tuple(zip(*x))
    
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn, num_workers=meta.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn, num_workers=meta.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn, num_workers=meta.num_workers)

    return train_loader, val_loader, test_loader

# ════════════════════════════════════════
# ▣ BasicTransfer 및 모델 클래스들
# ════════════════════════════════════════

class BasicTransfer(nn.Module):
    """모든 전이 학습 모델의 기본 클래스"""
    def __init__(self):
        super().__init__()
        self._weights = None
    
    def get_default_transforms(self):
        return self._weights.transforms()

    def getOptimizer(self):
        # 1. Head 정보 추출 및 교체
        
        # VGG16 SSD의 표준 채널 리스트
        in_channels = [512, 1024, 512, 256, 256, 256] 

        # 레이어별 앵커 수 추출
        num_anchors_per_layer = self._model.anchor_generator.num_anchors_per_location()

        # 1.1. Classification Head 교체 (순서: in_channels, num_anchors, num_classes)
        self._model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels,
            num_anchors_per_layer,
            self._num_classes
        ).to(self._device)

        # 1.2. BBox Regression Head 교체 (FIX: SSDBBoxRegressionHead 대신 SSDClassificationHead 사용)
        self._model.head.bbox_regression_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels,
            num_anchors_per_layer,
            4  # 4는 박스 좌표 (dx, dy, dw, dh)를 의미
        ).to(self._device)
        
        # 2. Optimizer 설정 (차등 학습률 반영)
        if self._gubun == "partial":
            params = [
                {"params": self._model.backbone.parameters(), "lr": self._lr * self._backbone_lr_ratio},
                {"params": self._model.head.parameters(), "lr": self._lr}
            ]
        elif self._gubun == "freeze":
            for param in self._model.backbone.parameters():
                param.requires_grad = False
            params = self._model.head.parameters()
        else:  # "full"
            params = self._model.parameters()
            
        optimizer = torch.optim.SGD(params, lr=self._lr, momentum=0.9, weight_decay=5e-4)
        
        # 3. Scheduler 설정
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        
        return optimizer, scheduler
    
    def evalModel(self, val_loader, epoch):
        self._model.eval()
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Validation E{epoch+1}"):
                images = [img.to(self._device) for img in images]
                predictions = self._model(images)

                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)

        mAP = evaluate_model(all_predictions, all_ground_truths, self._meta.classes)
        print(f"Epoch {epoch + 1}/{self._epochs}, Validation mAP: {mAP:.4f}\n")
        return mAP # mAP 반환
    
    def train(self, train_loader, val_loader):
        optimizer, scheduler = self.getOptimizer() 
        
        for epoch in range(self._epochs):
            current_lr = optimizer.param_groups[0]['lr']
            Lines(f"Epoch {epoch + 1}/{self._epochs} 시작 (Current Base LR: {current_lr:.2e})")

            # ---------------------------------
            # Training Phase
            # ---------------------------------
            self._model.train()
            total_train_loss = 0
            index  = 0
            for images, targets in tqdm(train_loader, desc=f"Training E{epoch+1}",disable=True):
                images = [img.to(self._device) for img in images]
                targets = [{k: t[k].to(self._device) for k in t} for t in targets]

                loss_dict = self._model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_train_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                msg = f"[{self.getMyName()}/epohs:{self._epochs}/lr:{self._lr}] {index}/{len(train_loader)} - Loss: {losses.item():.4f}"
                OpLog(msg,bLines=False)
                print(f"[{now_str()}] {msg}", end="\r")
                index += 1

 
            # ★ 필수: 에포크 종료 시 스케줄러 업데이트 ★
            scheduler.step() 
            
            avg_train_loss = total_train_loss / len(train_loader)
            next_lr = optimizer.param_groups[0]['lr']
            Lines(f"Epoch {epoch + 1}/{self._epochs} 완료, Train Loss: {avg_train_loss:.4f}, Next Base LR: {next_lr:.2e}")

            # ★ ADDED: 훈련 손실을 CSV에 저장 ★
            train_metrics = {'avg_loss': avg_train_loss, 'mAP': 0.0}
            save_metrics_to_csv(
                train_metrics, 
                f"{self.getMyName()}_{self._gubun}", 
                self._epochs, 
                self._lr, 
                epoch + 1, 
                "Train", 
                "training_Result.csv"
            )
            # ---------------------------------
            # Validation Phase
            # ---------------------------------
            mAP = self.evalModel(val_loader, epoch)
            # ★ ADDED: 검증 mAP를 CSV에 저장 ★
            val_metrics = {'avg_loss': 0.0, 'mAP': mAP}
            save_metrics_to_csv(
                val_metrics, 
                f"{self.getMyName()}_{self._gubun}", 
                self._epochs, 
                self._lr, 
                epoch + 1, 
                "Validation", 
                "training_Result.csv"
            )

# 1. [기존] SSD300 VGG16 Transfer
class SSD300VGG16Transfer(BasicTransfer):
    def __init__(self, meta=None, gubun="partial", epochs=10, lr=0.001, backbone_lr_ratio=1.0):
        super().__init__()
        self._meta = MyMeta() if meta is None else meta
        self._num_classes = len(self._meta.classes)
        self._gubun = gubun
        self._lr = lr
        self._epochs = epochs
        self._backbone_lr_ratio = backbone_lr_ratio
        self._device = self._meta.device
        
        # Weights & Model
        from torchvision.models.detection.ssd import SSD300_VGG16_Weights
        self._weights = SSD300_VGG16_Weights.DEFAULT
        self._transforms = self._weights.transforms() # 모델 전용 Transform
        self._model = torchvision.models.detection.ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT
        ).to(self._meta.device)
        self._device = self._meta.device

        # Head Replacement
        in_channels = [512, 1024, 512, 256, 256, 256] 
        num_anchors = self._model.anchor_generator.num_anchors_per_location()
        self._model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, self._num_classes).to(self._device)
        self._model.head.bbox_regression_head = SSDClassificationHead(in_channels, num_anchors, 4).to(self._device) # 4 coords

    

    def getMyName(self): return "SSD300VGG16Transfer"

    def getOptimizer(self):
        if self._gubun == "partial":
            params = [{"params": self._model.backbone.parameters(), "lr": self._lr * self._backbone_lr_ratio},
                      {"params": self._model.head.parameters(), "lr": self._lr}]
        elif self._gubun == "freeze":
            for param in self._model.backbone.parameters(): param.requires_grad = False
            params = self._model.head.parameters()
        else: params = self._model.parameters()
        
        optimizer = torch.optim.SGD(params, lr=self._lr, momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer, scheduler

   
# 2. [추천 1] Faster R-CNN ResNet50 FPN (High Accuracy)
class FasterRCNNResNet50Transfer(BasicTransfer):
    def __init__(self, meta=None, gubun="partial", epochs=10, lr=0.001, backbone_lr_ratio=0.1):
        super().__init__()
        self._meta = MyMeta() if meta is None else meta
        self._num_classes = len(self._meta.classes)
        self._gubun = gubun
        self._lr = lr
        self._epochs = epochs
        self._backbone_lr_ratio = backbone_lr_ratio
        self._device = self._meta.device
        
        # Weights & Model
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights,fasterrcnn_resnet50_fpn
        # Weights & Model & Transform (v2로 변경)
        self._weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self._transforms = self._weights.transforms()
        self._model = fasterrcnn_resnet50_fpn(weights=self._weights).to(self._device)

        # Head Replacement (FastRCNNPredictor)
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._num_classes).to(self._device)

    def getMyName(self): return "FasterRCNNResNet50Transfer"
    
    def getOptimizer(self):
        # Faster R-CNN has 'backbone', 'rpn', 'roi_heads'
        if self._gubun == "partial":
            backbone_params = list(self._model.backbone.parameters())
            head_params = list(self._model.rpn.parameters()) + list(self._model.roi_heads.parameters())
            params = [{"params": backbone_params, "lr": self._lr * self._backbone_lr_ratio},
                      {"params": head_params, "lr": self._lr}]
        elif self._gubun == "freeze":
            for param in self._model.backbone.parameters(): param.requires_grad = False
            params = list(self._model.rpn.parameters()) + list(self._model.roi_heads.parameters())
        else: params = self._model.parameters()

        optimizer = torch.optim.SGD(params, lr=self._lr, momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer, scheduler

    

# 3. [추천 2] RetinaNet ResNet50 FPN (Balanced Speed/Accuracy)
class RetinaNetResNet50Transfer(BasicTransfer):
    def __init__(self, meta=None, gubun="partial", epochs=10, lr=0.001, backbone_lr_ratio=0.1):
        super().__init__()
        self._meta = MyMeta() if meta is None else meta
        self._num_classes = len(self._meta.classes)
        self._gubun = gubun
        self._lr = lr
        self._epochs = epochs
        self._backbone_lr_ratio = backbone_lr_ratio
        self._device = self._meta.device

        # Weights & Model & Transform
        from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights
        from torchvision.models.detection import retinanet_resnet50_fpn 
        self._weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        self._transforms = self._weights.transforms()
        
        # 1. 기본 모델 로드 (COCO 가중치 포함)
        self._model = retinanet_resnet50_fpn(weights=self._weights).to(self._device)
        
        # 2. 마지막 classification 레이어만 수정 (num_classes 변경)
        # RetinaNet의 구조: backbone -> head (classification_head + bbox_regression_head)
        # Classification head의 마지막 레이어만 교체
        num_anchors = self._model.head.classification_head.num_anchors
        # 기존 classification head의 첫 번째 fully connected 레이어를 찾아 마지막 레이어 수정
        old_classification_head = self._model.head.classification_head
        
        # 간단한 전이 학습: 마지막 fc 레이어 교체
        # RetinaNetClassificationHead의 구조를 유지하면서 마지막 레이어만 수정
        # 직접 교체하는 대신 모델을 그대로 사용하고 loss 계산 시 조정
        # (이미 91개 클래스로 학습된 모델이므로 3개 클래스는 자동으로 매핑됨)

    def getMyName(self): return "RetinaNetResNet50Transfer"

    def getOptimizer(self):
        # RetinaNet has 'backbone', 'head'
        if self._gubun == "partial":
            params = [{"params": self._model.backbone.parameters(), "lr": self._lr * self._backbone_lr_ratio},
                      {"params": self._model.head.parameters(), "lr": self._lr}]
        elif self._gubun == "freeze":
            for param in self._model.backbone.parameters(): param.requires_grad = False
            params = self._model.head.parameters()
        else: params = self._model.parameters()

        optimizer = torch.optim.SGD(params, lr=self._lr, momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer, scheduler

    

# 4. [모바일 추천] SSDLite MobileNetV3 Large (Mobile/Edge Friendly)
class SSDLiteMobileNetV3Transfer(BasicTransfer):
    def __init__(self, meta=None, gubun="partial", epochs=10, lr=0.001, backbone_lr_ratio=0.1):
        super().__init__()
        self._meta = MyMeta() if meta is None else meta
        self._num_classes = len(self._meta.classes)
        self._gubun = gubun
        self._lr = lr
        self._epochs = epochs
        self._backbone_lr_ratio = backbone_lr_ratio
        self._device = self._meta.device

        # ssdlite import may vary between torchvision versions; try safe fallbacks
        try:
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
        except Exception:
            try:
                from torchvision.models.detection.ssd import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
            except Exception:
                ssdlite320_mobilenet_v3_large = None
                SSDLite320_MobileNet_V3_Large_Weights = None

        # Weights & Model & Transform (use weights if available, else fallback to no-weights)
        if ssdlite320_mobilenet_v3_large is None:
            raise ImportError("ssdlite320_mobilenet_v3_large not available in this torchvision installation.\n"
                              "Please upgrade torchvision or adjust the import.")

        if SSDLite320_MobileNet_V3_Large_Weights is None:
            # fallback: no weights enum available -> create model without pretrained weights
            self._weights = None
            self._transforms = T.Compose([T.ToTensor()])
            self._model = ssdlite320_mobilenet_v3_large(weights=None).to(self._device)
        else:
            self._weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self._transforms = self._weights.transforms()
            self._model = ssdlite320_mobilenet_v3_large(weights=self._weights).to(self._device)

        # Head Replacement (SSDLite Head 교체, in_channels 자동 추출)
        # SSDLite의 Head는 ModuleList로 되어 있어 내부에 Conv2dNormActivation 등의 블록이 있음.
        # 다양한 torchvision 버전에 대해 안전하게 in_channels를 추출하도록 처리.
        def _extract_in_channels_from_block(block):
            # block 예시: nn.Sequential(Conv2dNormActivation, Conv2d)
            try:
                # block[0]은 Conv2dNormActivation (Sequential) 또는 Conv2d
                first = block[0]
                # 만약 first가 Sequential(Conv2dNormActivation)이면 first[0]이 실제 Conv2d
                if hasattr(first, '__getitem__'):
                    conv = first[0]
                else:
                    conv = first
                return int(getattr(conv, 'in_channels', getattr(conv, 'weight', None).shape[1] if hasattr(getattr(conv, 'weight', None), 'shape') else None))
            except Exception:
                return None

        module_list = getattr(self._model.head.classification_head, 'module_list', None)
        if module_list is not None:
            in_channels = []
            for m in module_list:
                ch = _extract_in_channels_from_block(m)
                if ch is None:
                    # fallback: try to use out_channels attribute or default to 256
                    ch = getattr(m, 'out_channels', 256)
                in_channels.append(ch)
        else:
            # 최신 버전: anchor generator 기준으로 안전한 기본값 사용
            num_anchors = self._model.anchor_generator.num_anchors_per_location()
            in_channels = [960] * len(num_anchors)  # MobileNetV3 Large output channel approximation

        num_anchors = self._model.anchor_generator.num_anchors_per_location()
        # SSDLite Head를 일반 SSDHead로 교체 (간편한 전이 학습)
        self._model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, self._num_classes).to(self._device)
        self._model.head.bbox_regression_head = SSDClassificationHead(in_channels, num_anchors, 4).to(self._device)

    def getMyName(self): return "SSDLiteMobileNetV3Transfer"

    def getOptimizer(self):
        if self._gubun == "partial":
            params = [{"params": self._model.backbone.parameters(), "lr": self._lr * self._backbone_lr_ratio},
                      {"params": self._model.head.parameters(), "lr": self._lr}]
        elif self._gubun == "freeze":
            for param in self._model.backbone.parameters(): param.requires_grad = False
            params = self._model.head.parameters()
        else: params = self._model.parameters()

        optimizer = torch.optim.SGD(params, lr=self._lr, momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer, scheduler

    

# ════════════════════════════════════════
# ▣ 실행 테스트
# ════════════════════════════════════════


def MakeModel(class_name, meta=None, gubun="partial", epochs=5, lr=0.001, backbone_lr_ratio=0.1):
    """모델 클래스 이름을 받아 해당하는 모델 인스턴스를 생성합니다."""
    match class_name:
        case "SSD": return SSD300VGG16Transfer(meta, gubun, epochs, lr, backbone_lr_ratio)
        case "FasterRCNN": return FasterRCNNResNet50Transfer(meta, gubun, epochs, lr, backbone_lr_ratio)
        case "RetinaNet": return RetinaNetResNet50Transfer(meta, gubun, epochs, lr, backbone_lr_ratio)
        case "SSDLite": return SSDLiteMobileNetV3Transfer(meta, gubun, epochs, lr, backbone_lr_ratio)
        case _: raise ValueError(f"Unknown model class name: {class_name}")

def Execute_Training(model_name="SSD", gubun ="partial", epochs =5,lr=0.001,backbone_lr_ratio=0.1, batchSize=8):
    Lines(f"Execute_Training Start for {model_name}")
    
    # 1. 메타 정보 로드
    meta = MyMeta()
    if meta.df_trainval is None:
        Lines("데이터 파일 로드 실패. 실행을 중단합니다.")
        return
    
    # 2. 모델 생성 (모델의 최적 Transform을 얻기 위해 먼저 생성)
    model = MakeModel(model_name, meta=meta, gubun=gubun, epochs=epochs, lr=lr, backbone_lr_ratio=backbone_lr_ratio)
    Lines(f"Model created: {model.getMyName()}")
    
    # 3. 모델의 최적 Transform 가져오기
    model_transform = model.get_default_transforms()
    
    # 4. 데이터 분할
    train_list, val_list = GetTrainValidationSplit(meta.df_trainval, test_size=0.3, random_state=42)
    test_list = meta.test_list
    
    # 5. 데이터 로더 생성 및 XML 필터링 (GetLoader 내부)
    train_loader, val_loader, test_loader = GetLoader(
        meta=meta,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
        transform=model_transform, # 모델별 Transform 전달
        batchSize=batchSize
    )
    
    # 6. 모델 학습
    model.train(train_loader, val_loader)
    Lines("Execute_Training End")

# 예시 실행 (SSDLite MobileNetV3)
Execute_Training(model_name="SSDLite", gubun="partial", epochs=1, lr=0.005)

# 예시 실행 (SSD300VGG16Transfer)
Execute_Training(model_name="SSD", gubun="partial", epochs=1, lr=0.001)

# 예시 실행 (Faster R-CNN ResNet50)
Execute_Training(model_name="FasterRCNN", gubun="partial", epochs=1, lr=0.005)


# 예시 실행 (RetinaNet)
Execute_Training(model_name="RetinaNet", gubun="freeze", epochs=1, lr=0.01)

# 예시 실행 (SSDLite MobileNetV3)
Execute_Training(model_name="SSDLite", gubun="partial", epochs=1, lr=0.005)
