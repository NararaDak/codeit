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
import matplotlib
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
import matplotlib.patches as patches
from filelock import FileLock
import yaml

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDClassificationHead

# ════════════════════════════════════════
# ▣ 유틸리티
# ════════════════════════════════════════
BASE_DIR = r"D:\05.gdrive\codeit\mission7\data\pet_data" 
BASE_DIR = r"D:\01.project\CodeIt\mission7\data\pet_data" 
#BASE_DIR = "/content/drive/MyDrive/codeit/mission7/data/pet_data" # Colab용
RESULT_CSV = f"{BASE_DIR}/training_Result.csv"

def Lines(text="", count=100):
    print("=" * count)
    if text != "":
        print(f"{text}")
        print("=" * count)

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
        self._modelfiles_dir = os.path.join(self._base_dir, "modelfiles")
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
    @property
    def modelfiles_dir(self): return self._modelfiles_dir
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

def save_metrics_to_csv(metrics, params_name, epochs, learnRate, epoch_index, data_set_name):
    mAP = metrics.get('mAP', 0.0)
    new_data = {
        'Strategy': [params_name], 'Max_Epochs': [epochs], 'Epoch_Index': [epoch_index],
        'DataSet': [data_set_name], 'LearnRate': [learnRate], 'AvgLoss': [metrics.get('avg_loss', 0.0)],
        'Accuracy': [mAP], # mAP 저장
        'Precision': [0.0], 'Recall': [0.0], 'Specificity': [0.0], 'F1Score': [0.0],
        'TN': [0], 'FP': [0], 'FN': [0], 'TP': [0]
    }
    filename = RESULT_CSV
    lock_filename = filename + ".lock"
    new_df = pd.DataFrame(new_data)
    
    try:
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
    # print(f"Metrics saved: {data_set_name} Epoch {epoch_index}")

def GetTrainValidationSplit(df, test_size=0.3, random_state=42):
    meta = MyMeta()
    trainval_list = meta.trainval_list
    train_list, valid_list = train_test_split(trainval_list, test_size=0.3, random_state=42)
    Lines(f"Train/Validation :{len(train_list)},{len(valid_list)}")
    return train_list, valid_list

Lines("Define Environment")


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
    def __init__(self, image_dir, image_list, classes, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = image_list
        self.classes = classes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx] + ".jpg"
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        
        # 테스트셋은 일반적으로 타겟이 없지만, 평가를 위해 가능하면 XML 주석에서 GT를 읽어 반환
        annotation_dir = os.path.join(os.path.dirname(self.image_dir), 'annotations', 'annotations', 'xmls')
        annotation_file = os.path.join(annotation_dir, os.path.splitext(image_file)[0] + '.xml')

        boxes = []
        labels = []
        if os.path.exists(annotation_file):
            try:
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    # 클래스 매핑은 `self.classes` (MyMeta.classes)를 사용
                    if class_name in self.classes:
                        cls_idx = self.classes.index(class_name)
                    else:
                        # fallback: try common names
                        cls_idx = 1 if class_name.lower() in ('dog',) else 2 if class_name.lower() in ('cat',) else 0

                    bndbox = obj.find('bndbox')
                    if bndbox is None:
                        continue
                    boxes.append([
                        int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)
                    ])
                    labels.append(cls_idx)
            except Exception:
                # parsing 실패 시 빈 GT 사용
                boxes = []
                labels = []
        else:
            # 주석 파일이 없으면 빈 GT 사용
            boxes = []
            labels = []

        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image) # 테스트셋은 이미지만 변환

        return image, target
Lines("Define dataset")

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
    test_dataset = TestDataset(meta.image_dir, meta.test_list, meta.classes, transforms=transform)
    
    collate_fn = lambda x: tuple(zip(*x))
    
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn, num_workers=meta.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn, num_workers=meta.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn, num_workers=meta.num_workers)

    return train_loader, val_loader, test_loader

def GetTestLoader(meta, transform, batchSize=8):
    test_dataset = TestDataset(meta.image_dir, meta.test_list, meta.classes, transforms=transform)
    collate_fn = lambda x: tuple(zip(*x))
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn, num_workers=meta.num_workers)
    return test_loader
Lines("Dataset and DataLoader defined.")


# ════════════════════════════════════════
# ▣ BasicTransfer 및 모델 클래스들
# ════════════════════════════════════════
class BasicTransfer(nn.Module):
    """모든 전이 학습 모델의 기본 클래스"""
    def __init__(self):
        super().__init__()
        self._weights = None
        self._transforms = None
    
    def get_default_transforms(self):
        return self._transforms

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
    def visualize_prediction(self,image, prediction, classes):
        # """
        # image (torch.Tensor): 추론에 사용된 이미지 (C, H, W 형식).
        # prediction (dict): 모델의 예측 결과 (boxes, labels, scores 포함).
        # classes (list): 클래스 이름 리스트.
        # """
        # Tensor 이미지를 (H, W, C) 형식으로 변환
        image = image.permute(1, 2, 0).numpy()

        # Matplotlib을 사용한 이미지 시각화
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        # Bounding Box와 클래스 이름 시각화
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            if score > 0.5:  # Confidence Score 임계값
                x_min, y_min, x_max, y_max = box.tolist()
                width, height = x_max - x_min, y_max - y_min

                # Bounding Box 추가
                rect = patches.Rectangle(
                    (x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)

                # 클래스 이름과 Confidence Score 추가
                ax.text(
                    x_min,
                    y_min - 10,
                    f"{classes[label]}: {score:.2f}",
                    color="red",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7),
                )

        plt.axis("off")
        # Do not use interactive display functions (plt.pause/plt.show) to avoid Tcl/Tk threading issues.
        # We simply close the figure; saving of examples is handled by `save_Image` which uses Agg backend.
        plt.close()

    def evalModel(self, valloader, epoch, save_images=True, max_images=5):
        """Run evaluation on `valloader`, compute mAP and optionally save example prediction images.
        - `save_images`: whether to save example prediction images using `save_Image`.
        - `max_images`: maximum number of images to collect and save.
        """
        self._model.eval()
        all_predictions = []
        all_ground_truths = []
        images_for_saving = []

        with torch.no_grad():
            for images, targets in tqdm(valloader, desc=f"Validation E{epoch+1}"):
                # PIL Image를 텐서로 변환 (필요시)
                if isinstance(images[0], Image.Image):
                    images = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 if isinstance(img, Image.Image) else img for img in images]
                
                # SSD는 리스트 형식의 이미지를 받음 (다양한 크기 허용)
                images = [img.to(self._device) if isinstance(img, torch.Tensor) else img for img in images]
                predictions = self._model(images)

                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)

                # 수집된 예시 이미지를 최대 `max_images`개까지 저장용으로 보관
                if save_images and len(images_for_saving) < max_images:
                    for i, (img, pred) in enumerate(zip(images, predictions)):
                        if len(images_for_saving) >= max_images:
                            break
                        try:
                            images_for_saving.append((img.cpu(), {k: v.cpu() for k, v in pred.items()}))
                        except Exception as e:
                            # 예외가 발생하면 건너뜀
                            OpLog(f"Warning: Failed to add image {i} to save list in evalModel: {e}", bLines=False)
                            continue

        mAP = evaluate_model(all_predictions, all_ground_truths, self._meta.classes)
        print(f"Epoch {epoch + 1}/{self._epochs}, Validation mAP: {mAP:.4f}\n")

        # 마지막 배치의 첫 이미지/예측이 존재하면 시각화 (non-blocking)
        try:
            if 'images' in locals() and len(images) > 0 and len(predictions) > 0:
                self.visualize_prediction(images[0].cpu(), predictions[0], self._meta.classes)
        except Exception:
            pass

        # 저장 옵션이 켜져 있으면 예시 이미지들을 저장 폴더에 기록
        try:
            if save_images and images_for_saving:
                OpLog(f"Saving {len(images_for_saving)} validation images...", bLines=False)
                self.save_Image(True, images_for_saving, max_images, epoch, mode="Validation")
                OpLog(f"✓ Validation images saved successfully", bLines=False)
            elif save_images:
                OpLog(f"Warning: No images collected for saving (images_for_saving is empty)", bLines=False)
        except Exception as e:
            OpLog(f"Error saving validation images: {e}", bLines=False)
            import traceback
            OpLog(traceback.format_exc(), bLines=False)

        return mAP # mAP 반환

    def testModel(self, test_loader, epoch_index=0, save_images=True, max_images=10):
        """Run inference on test_loader, compute mAP, save metrics to CSV and save example prediction images.
        Images saved under: <modelfiles_dir>/<modelName>/epochs_<epochs>/epoch_<epoch_index>/lr_<lr>/
        Up to `max_images` images will be saved.
        """
        self._model.eval()
        all_predictions = []
        all_ground_truths = []
        images_for_saving = []

        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc=f"Test"):
                images = [img.to(self._device) for img in images]
                predictions = self._model(images)

                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)

                # collect examples for saving (move tensors to cpu)
                if save_images and len(images_for_saving) < max_images:
                    for i, (img, pred) in enumerate(zip(images, predictions)):
                        if len(images_for_saving) >= max_images:
                            break
                        try:
                            images_for_saving.append((img.cpu(), {k: v.cpu() for k, v in pred.items()}))
                        except Exception as e:
                            OpLog(f"Warning: Failed to add test image {i} to save list: {e}", bLines=False)
                            continue

        mAP = evaluate_model(all_predictions, all_ground_truths, self._meta.classes)
        save_metrics_to_csv({'avg_loss': 0.0, 'mAP': mAP}, f"{self.getMyName()}_{self._gubun}", self._epochs, self._lr, epoch_index, "Test")
        
        # 이미지 저장
        try:
            if save_images and images_for_saving:
                OpLog(f"Saving {len(images_for_saving)} test images...", bLines=False)
                self.save_Image(True, images_for_saving, max_images, epoch_index, mode="Test")
                OpLog(f"✓ Test images saved successfully", bLines=False)
            elif save_images:
                OpLog(f"Warning: No images collected for saving (images_for_saving is empty)", bLines=False)
        except Exception as e:
            OpLog(f"Error saving test images: {e}", bLines=False)
            import traceback
            OpLog(traceback.format_exc(), bLines=False)
        
        print(f"Test mAP: {mAP:.4f}")
        return mAP
    def save_Image(self,save_images,images_for_saving, max_images, epoch_index,mode):
        # save metrics to CSV

        # save images
        if save_images and images_for_saving:
            base_dir = Path(self._meta.modelfiles_dir)
            save_dir = base_dir / f"{self.getMyName()}_{mode}_{self._gubun}_{self._epochs}_{epoch_index}_{self._lr}"
            os.makedirs(save_dir, exist_ok=True)
            OpLog(f"Saving images to: {save_dir}", bLines=False)
            
            saved_count = 0
            for idx, (img_tensor, pred) in enumerate(images_for_saving[:max_images]):
                try:
                    # 이미지 텐서를 numpy 배열로 변환
                    if isinstance(img_tensor, torch.Tensor):
                        img = img_tensor.permute(1, 2, 0).cpu().numpy()
                    else:
                        img = img_tensor
                    
                    # Normalize/scale to 0-255 if needed
                    if img.max() <= 1.0:
                        img = (img * 255.0).astype('uint8')
                    else:
                        img = img.astype('uint8')

                    fig, ax = plt.subplots(1, figsize=(8, 8))
                    ax.imshow(img)
                    
                    # 예측 박스 그리기
                    boxes = pred['boxes'] if isinstance(pred['boxes'], torch.Tensor) else torch.tensor(pred['boxes'])
                    labels = pred['labels'] if isinstance(pred['labels'], torch.Tensor) else torch.tensor(pred['labels'])
                    scores = pred['scores'] if isinstance(pred['scores'], torch.Tensor) else torch.tensor(pred['scores'])
                    
                    for box, label, score in zip(boxes, labels, scores):
                        if score > 0.5:
                            if isinstance(box, torch.Tensor):
                                x_min, y_min, x_max, y_max = box.cpu().tolist()
                            else:
                                x_min, y_min, x_max, y_max = box
                            width, height = x_max - x_min, y_max - y_min
                            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='red', facecolor='none')
                            ax.add_patch(rect)
                            label_int = int(label) if isinstance(label, torch.Tensor) else int(label)
                            score_float = float(score) if isinstance(score, torch.Tensor) else float(score)
                            ax.text(x_min, y_min - 10, f"{self._meta.classes[label_int]}: {score_float:.2f}", color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
                    plt.axis('off')
                    out_path = save_dir / f"pred_{idx+1}.png"
                    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    saved_count += 1
                except Exception as e:
                    OpLog(f"Error saving image {idx+1}: {e}", bLines=False)
                    import traceback
                    OpLog(traceback.format_exc(), bLines=False)
                    continue
            
            OpLog(f"Successfully saved {saved_count}/{len(images_for_saving[:max_images])} images to {save_dir}", bLines=False)
        else:
            if not save_images:
                OpLog(f"Image saving is disabled (save_images=False)", bLines=False)
            elif not images_for_saving:
                OpLog(f"No images to save (images_for_saving is empty)", bLines=False)



    def saveCheckPoint(self,gubun, num_epochs,current_epoch_index, learnRate, model, optimizer, avg_train_loss):
        modelfile_dir = self._meta.modelfiles_dir
        checkpoint_path = Path(modelfile_dir) / f"checkpoint_{self.getMyName()}_{gubun}_{num_epochs}_{learnRate}_epoch_{current_epoch_index:02d}.pth"
        torch.save({
            'epoch': current_epoch_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'gubun': gubun,
            'learnRate': learnRate,
        }, checkpoint_path)
        OpLog(f"Checkpoint saved to '{checkpoint_path}'")
    # [추가/수정] 전체 학습 루프를 포함하는 train 메서드 구현 (과적합 방지 기법 포함)                                                                                   
    def train(self, train_loader, val_loader,test_loader):
        optimizer, scheduler = self.getOptimizer() 
        avg_train_loss = 0.0
        
        # ═══ 과적합 방지 설정 ═══
        patience = 5  # Early Stopping: 5 에포크 동안 개선 없으면 종료
        patience_counter = 0
        best_val_mAP = -float('inf')
        best_model_state = None
        
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
                # PIL Image를 텐서로 변환 (필요시)
                if isinstance(images[0], Image.Image):
                    images = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0 if isinstance(img, Image.Image) else img for img in images]
                
                # SSD는 리스트 형식의 이미지를 받음 (다양한 크기 허용)
                images = [img.to(self._device) if isinstance(img, torch.Tensor) else img for img in images]
                targets = [{k: t[k].to(self._device) for k in t} for t in targets]

                loss_dict = self._model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_train_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)  # Gradient Clipping
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
                "Train" 
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
                "Validation" 
            )
            
            # ═══ Early Stopping & Best Model 저장 ═══
            if mAP > best_val_mAP:
                best_val_mAP = mAP
                patience_counter = 0
                best_model_state = self._model.state_dict().copy()
                OpLog(f"✓ Best model updated: mAP={best_val_mAP:.4f}")
            else:
                patience_counter += 1
                OpLog(f"✗ No improvement for {patience_counter}/{patience} epochs. Best mAP: {best_val_mAP:.4f}")
            
            # Early Stopping 체크
            if patience_counter >= patience:
                OpLog(f"Early Stopping triggered at epoch {epoch + 1}. Loading best model...")
                if best_model_state is not None:
                    self._model.load_state_dict(best_model_state)
                break
            
            self.saveCheckPoint(self._gubun, self._epochs, epoch + 1, self._lr, self._model, optimizer, avg_train_loss)
            # 매 에포크마다 Validation 후 즉시 Test 호출로 결과 및 이미지 저장
            self.testModel(test_loader, epoch_index=epoch + 1)

# 1. [기존] SSD300 VGG16 Transfer
class SSD300VGG16Transfer(BasicTransfer):
    def __init__(self, meta=None, gubun="partial", epochs=10, lr=0.001, backbone_lr_ratio=1.0):
        super().__init__()
        self._meta = MyMeta() if meta is None else meta
        #self._num_classes = max(len(self._meta.classes) - 1, 1)
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

    def train(self, train_loader, val_loader, test_loader):
        """SSD300VGG16 전용 학습 루프"""
        return super().train(train_loader, val_loader, test_loader)
# . [모바일 추천] SSDLite MobileNetV3 Large (Mobile/Edge Friendly)
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
            """Safely extract in_channels from a block (Sequential with Conv2d or Conv2dNormActivation)."""
            try:
                # block 구조: Sequential(Conv2dNormActivation(...), Conv2d(...))
                # 또는 Sequential(Conv2d(...), ...)
                for module in block:
                    if isinstance(module, torch.nn.Conv2d):
                        return int(module.in_channels)
                    elif hasattr(module, '__iter__'):
                        # Nested Sequential (e.g., Conv2dNormActivation)
                        for sub_module in module:
                            if isinstance(sub_module, torch.nn.Conv2d):
                                return int(sub_module.in_channels)
                # Fallback: check weight shape
                if hasattr(block, 'weight'):
                    return int(block.weight.shape[1])
                return None
            except Exception:
                return None

        module_list = getattr(self._model.head.classification_head, 'module_list', None)
        if module_list is not None and len(module_list) > 0:
            in_channels = []
            for m in module_list:
                ch = _extract_in_channels_from_block(m)
                if ch is None:
                    # Fallback 1: try out_channels attribute
                    ch = getattr(m, 'out_channels', None)
                if ch is None:
                    # Fallback 2: try to infer from first Conv2d in module
                    try:
                        if hasattr(m, '__iter__'):
                            for sub_m in m:
                                if hasattr(sub_m, 'out_channels'):
                                    ch = sub_m.out_channels
                                    break
                    except Exception:
                        pass
                if ch is None:
                    # Fallback 3: default MobileNetV3 output channel
                    ch = 960
                in_channels.append(ch)
        else:
            # No module_list or empty: use anchor generator based defaults
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

    def train(self, train_loader, val_loader, test_loader):
        """SSDLite MobileNetV3 전용 학습 루프"""
        return super().train(train_loader, val_loader, test_loader)

# ultralytics 라이브러리 사용
from ultralytics import YOLO 
from ultralytics.nn.tasks import DetectionModel


# YOLOv8n Transfer
class YOLOv8nTransfer(BasicTransfer):
    def __init__(self, meta=None, gubun="partial", epochs=10, lr=0.01, backbone_lr_ratio=0.1):
        super().__init__()
        self._meta = MyMeta() if meta is None else meta
        self._num_classes = len(self._meta.classes)
        self._gubun = gubun
        self._lr = lr
        self._epochs = epochs
        self._backbone_lr_ratio = backbone_lr_ratio
        self._device = self._meta.device
        # YOLOv8n 가중치 초기화: 로컬 파일 우선, 없으면 ultralytics hub에서 다운로드
        self._yolo = None
        
        # 1단계: 여러 경로에서 로컬 yolov8n.pt 파일 검색
        weight_candidates = [
            'yolov8n.pt',  # 현재 디렉토리
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR))), 'yolov8n.pt'),  # 프로젝트 루트
            os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'yolov8n.pt')  # 상위 디렉토리
        ]
        
        # 로컬 파일이 존재하면 해당 파일로 YOLO 래퍼 초기화
        for w in weight_candidates:
            try:
                if os.path.exists(w):
                    OpLog(f"Loading YOLO weights from: {w}")
                    self._yolo = YOLO(w)
                    break
            except Exception:
                continue
        
        # 2단계: 로컬 파일 없으면 ultralytics가 자동으로 다운로드하도록 시도
        if self._yolo is None:
            try:
                OpLog("Loading YOLO by short name 'yolov8n.pt' (may use internet or local cache)")
                self._yolo = YOLO('yolov8n.pt')  # ultralytics hub에서 자동 다운로드
            except Exception as e:
                OpLog(f"Warning: failed to load ultralytics YOLO wrapper: {e}")
                self._yolo = None

        # 3단계: 래퍼 내부 모델을 self._model에 참조 (평가/추론 시 사용)
        try:
            if self._yolo is not None and hasattr(self._yolo, 'model'):
                self._model = self._yolo.model.to(self._device)
        except Exception:
            pass
    def getMyName(self):
        return "YOLOv8nTransfer"
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load YOLO checkpoint from .pt file.
        
        Args:
            checkpoint_path: Path to the YOLO checkpoint file (.pt)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            OpLog(f"Loading YOLO checkpoint: {checkpoint_path}")
            
            # Load YOLO model from checkpoint
            self._yolo = YOLO(checkpoint_path)
            
            # Update model reference
            if hasattr(self._yolo, 'model'):
                self._model = self._yolo.model.to(self._device)
                OpLog(f"Successfully loaded YOLO checkpoint from {checkpoint_path}")
                return True
            else:
                OpLog(f"Warning: YOLO wrapper has no 'model' attribute")
                return False
                
        except Exception as e:
            OpLog(f"Failed to load YOLO checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def PreJob(self):
        #  YOLO 학습용 데이터셋 YAML 파일 준비 (pets.yaml)
        #  기존 YAML이 있으면 재사용
        #  없으면 train/val 경로와 클래스 정보로 생성
        try:
            base = getattr(self._meta, 'base_dir', BASE_DIR)
            yaml_path = os.path.join(base, 'annotations', 'pets.yaml')

            # 기존 YAML 파일이 있으면 즉시 반환
            if os.path.exists(yaml_path):
                OpLog(f"PreJob: found existing YAML: {yaml_path}")
                return yaml_path

            # 이미지 디렉토리 결정 (train/val 하위폴더가 있으면 우선 사용)
            image_dir = getattr(self._meta, 'image_dir', os.path.join(base, 'images', 'images'))
            train_dir = os.path.join(image_dir, 'train') if os.path.isdir(os.path.join(image_dir, 'train')) else image_dir
            val_dir = os.path.join(image_dir, 'val') if os.path.isdir(os.path.join(image_dir, 'val')) else image_dir

            # 클래스 이름 추출 (background 제외)
            names = [c for c in getattr(self._meta, 'classes', []) if c.lower() != 'background']
            nc = len(names) if len(names) > 0 else 1

            # YAML 내용 생성 (YOLO 표준 포맷)
            content_lines = [f"train: {os.path.abspath(train_dir)}", f"val: {os.path.abspath(val_dir)}", f"nc: {nc}", "names:"]
            for i, n in enumerate(names):
                content_lines.append(f"  {i}: {n}")
            content = "\n".join(content_lines) + "\n"

            # 파일 저장
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(content)
            OpLog(f"PreJob: created minimal YAML at {yaml_path}")
            return yaml_path
        except Exception as e:
            OpLog(f"PreJob: failed to create/find YAML: {e}")
            return None
    def get_default_transforms(self):
        # YOLO requires fixed size 640x640 without normalization
        transforms = [
            T.Resize((640, 640)),
            T.ToTensor()
        ]
        return T.Compose(transforms)
    def train(self, train_loader, val_loader, test_loader):
        # ultralytics YOLO 래퍼가 있으면 고수준 API 사용
        if self._yolo is not None:
            # YAML 준비 (없으면 생성)
            yaml_path = self.PreJob()
            if yaml_path is None:
                OpLog("YOLO yaml not prepared; falling back to manual training loop")
                return super().train(train_loader, val_loader, test_loader)
            
            try:
                OpLog(f"Starting ultralytics YOLO.train with data={yaml_path}, epochs={self._epochs}")
                # ultralytics 고수준 API로 학습 (데이터 로더 불필요)
                results = self._yolo.train(
                    data=yaml_path,           # YAML 경로 (train/val 이미지 경로 포함)
                    epochs=self._epochs,      # 에포크 수
                    imgsz=640,                # 이미지 크기 (YOLO 표준)
                    name=f"{self.getMyName()}_{self._gubun}",  # 실험 이름
                    lr0=self._lr,             # 초기 학습률
                    device=self._device,      # GPU/CPU 디바이스
                    amp=True if torch.cuda.is_available() else False,  # 자동 혼합 정밀도
                )
                OpLog(f"YOLO.train completed: {results}")
                
                # ▶ 학습 완료 후 체크포인트 저장
                try:
                    OpLog("Saving YOLOv8n checkpoint...")
                    self.saveCheckPoint(
                        self._gubun, 
                        self._epochs, 
                        self._epochs,  # ▶ 마지막 에포크
                        self._lr, 
                        self._yolo.model if hasattr(self._yolo, 'model') else self._model, 
                        None,  # ▶ optimizer는 ultralytics 내부 관리
                        0.0    # ▶ avg_train_loss는 ultralytics에서 추출 불가
                    )
                except Exception as e:
                    OpLog(f"YOLOv8n checkpoint save failed: {e}", bLines=False)
                
                # ▶ 학습 완료 후 Test 데이터셋으로 mAP 평가 및 CSV 저장
                try:
                    OpLog("Running testModel to evaluate and save metrics to CSV...")
                    self.testModel(test_loader, epoch_index=self._epochs)
                except Exception as e:
                    OpLog(f"YOLOv8n post-train testModel failed: {e}", bLines=False)
                
                return results
            except Exception as e:
                OpLog(f"ultralytics YOLO.train failed: {e}")
                OpLog("Falling back to BasicTransfer.train()")
                return super().train(train_loader, val_loader, test_loader)
        
        # ultralytics 래퍼 없음 -> 기존 수동 학습 루프 사용
        OpLog("No ultralytics YOLO wrapper available; using BasicTransfer.train()")
        return super().train(train_loader, val_loader, test_loader)
        
Lines("Define Transformer classes")


def MakeModel(class_name, meta=None, gubun="partial", epochs=5, lr=0.001, backbone_lr_ratio=0.1):
    """모델 클래스 이름을 받아 해당하는 모델 인스턴스를 생성합니다."""
    match class_name:
        case "SSD": return SSD300VGG16Transfer(meta, gubun, epochs, lr, backbone_lr_ratio)
        case "SSDLite": return SSDLiteMobileNetV3Transfer(meta, gubun, epochs, lr, backbone_lr_ratio)
        case "YOLOv8n": return YOLOv8nTransfer(meta, gubun, epochs, lr, backbone_lr_ratio)
        case _: raise ValueError(f"Unknown model class name: {class_name}")

def Execute_Training_org(model_name="SSD", gubun ="partial", epochs =5,lr=0.001,backbone_lr_ratio=0.1, batchSize=8):
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
    model.train(train_loader, val_loader,test_loader)
    Lines("Execute_Training End")


def Execute_Training(model_name="YOLOv8n", gubun="partial", epochs=20, lr=0.01, backbone_lr_ratio=0.05, batchSize=16):
    Lines(f"Execute_Training Start for {model_name}")
    Lines(f"Parameters: gubun={gubun}, epochs={epochs}, lr={lr}, backbone_lr_ratio={backbone_lr_ratio}, batchSize={batchSize}")

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
    model.train(train_loader, val_loader, test_loader)
    Lines("Execute_Training End")
Lines("Define Transformer classes End")

# ════════════════════════════════════════
# ▣ 실행 예시 (모델별 권장 옵션)
# ════════════════════════════════════════
def Trains():

    # 2. SSD300 VGG16 (균형잡힌 정확도/속도)
    # 권장: lr=0.001, epochs=10, backbone_lr_ratio=1.0, batchSize=8 (전체 또는 부분 전이학습)
    Execute_Training(model_name="SSD", gubun="partial", epochs=10, lr=0.001, backbone_lr_ratio=1.0, batchSize=8)

    # 1. SSDLite MobileNetV3 (Mobile-friendly, 빠름)
    # 권장: lr=0.001, epochs=30, backbone_lr_ratio=0.1, batchSize=16 (긴 학습으로 안정화)
    Execute_Training(model_name="SSDLite", gubun="partial", epochs=30, lr=0.001, backbone_lr_ratio=0.1, batchSize=16)
 
    # 5. YOLOv8n (빠르고 정확함)
    # 권장(권장 기본): lr=0.01, epochs=20, backbone_lr_ratio=0.05, batchSize=16
    # - YOLO은 비교적 큰 lr(예: 0.01)과 AMP 사용으로 빠르게 수렴합니다.
    Execute_Training(model_name="YOLOv8n", gubun="partial", epochs=20, lr=0.01, backbone_lr_ratio=0.05, batchSize=16)
# Trains()
Lines("학습")


# ════════════════════════════════════════
# ▣ 모델 로드 및 추론.
# ════════════════════════════════════════
# 모델 로드.ArithmeticError
def visualize_prediction(image, prediction, classes):
    """
    image (torch.Tensor): 추론에 사용된 이미지 (C, H, W 형식).
    prediction (dict): 모델의 예측 결과 (boxes, labels, scores 포함).
    classes (list): 클래스 이름 리스트.
    save_path (str, optional): 이미지를 저장할 경로. None이면 저장 안 함.
    display_seconds (float): 화면에 표시할 시간(초). 0이면 표시 안 함. 음수면 무한대로 유지.
    """
    # Tensor 이미지를 (H, W, C) 형식으로 변환
    image = image.permute(1, 2, 0).numpy()
    
    # 0~1 범위 정규화 처리
    if image.max() <= 1.0:
        image = (image * 255.0).astype('uint8')
    else:
        image = image.astype('uint8')

    # Matplotlib을 사용한 이미지 시각화
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Bounding Box와 클래스 이름 시각화
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score > 0.5:  # Confidence Score 임계값
            x_min, y_min, x_max, y_max = box.tolist()
            width, height = x_max - x_min, y_max - y_min

            # Bounding Box 추가
            rect = patches.Rectangle(
                (x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

            # 클래스 이름과 Confidence Score 추가
            ax.text(
                x_min,
                y_min - 10,
                f"{classes[int(label)]}: {score:.2f}",
                color="red",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )
    plt.axis("off")
    plt.show()
    plt.imshow(image)
    plt.close(fig)

def LoadMode(file, device):
    """Load checkpoint and return state_dict only. Handle multiple checkpoint formats."""
    try:
        # Try with weights_only=False for old checkpoints with numpy objects
        if torch.cuda.is_available():
            checkpoint = torch.load(file, weights_only=False)
        else:
            checkpoint = torch.load(file, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=False: {e}")
        # Fallback to default loading
        if torch.cuda.is_available():
            checkpoint = torch.load(file)
        else:
            checkpoint = torch.load(file, map_location=device)
    
    # Handle multiple checkpoint formats
    if isinstance(checkpoint, dict):
        # Format 1: {'model_state_dict': {...}, 'epoch': ..., 'optimizer_state_dict': ...}
        if 'model_state_dict' in checkpoint:
            print(f"Loaded checkpoint with model_state_dict key")
            return checkpoint['model_state_dict']
        
        # Format 2: {'model_state': {...}, ...} (alternative naming)
        if 'model_state' in checkpoint:
            print(f"Loaded checkpoint with model_state key")
            return checkpoint['model_state']
        
        # Format 3: Direct state_dict (no wrapper keys) - check by looking for typical model keys
        # Look for keys like 'backbone.*', 'head.*' which indicate this is already a state_dict
        if any(key.startswith(('backbone', 'head')) for key in checkpoint.keys()):
            print(f"Loaded checkpoint as direct state_dict")
            return checkpoint
        
        # If all keys look foreign, try to find the largest dict inside (which might be the model)
        dict_values = [v for v in checkpoint.values() if isinstance(v, dict)]
        if dict_values:
            largest_dict = max(dict_values, key=len)
            if any(key.startswith(('backbone', 'head')) for key in largest_dict.keys()):
                print(f"Found embedded state_dict in checkpoint")
                return largest_dict
    
    print(f"Warning: Could not identify state_dict structure, returning checkpoint as-is")
    return checkpoint

def testModel(meta, model, test_loader, max_images=10):
        """Run inference on test_loader, compute mAP, save metrics to CSV and save example prediction images.
        Images saved under: <modelfiles_dir>/<modelName>/epochs_<epochs>/epoch_<epoch_index>/lr_<lr>/
        Up to `max_images` images will be saved.
        Args:
            meta: MyMeta instance
            model: nn.Module model instance (must have .eval() and be callable)
            test_loader: DataLoader for test data
            max_images: max number of images to save
        """
        device = meta.device
        model = model.to(device)
        model.eval()
        all_predictions = []
        all_ground_truths = []
        images_for_saving = []


        with torch.no_grad():
            nCount = 0
            for images, targets in tqdm(test_loader, desc=f"Test"):
                images = [img.to(device) for img in images]
                predictions = model(images)  # Call model directly
                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)
                # collect examples for saving (move tensors to cpu)
                if len(images_for_saving) < max_images:
                    for img, pred in zip(images, predictions):
                        if len(images_for_saving) >= max_images:
                            break
                        images_for_saving.append((img.cpu(), {k: v.cpu() for k, v in pred.items()}))    
                nCount += len(images)
                if nCount >= max_images:
                    break
            # Visualize last batch if available
            if 'images' in locals() and len(images) > 0 and len(predictions) > 0:
                try:
                    for idx, (img, pred) in enumerate(zip(images, predictions)):
                        # Save with timestamp and show for 3 seconds
                        save_path = f"pred_sample_{idx+1}.png"
                        visualize_prediction(
                            img.cpu(), 
                            {k: v.cpu() for k, v in pred.items()}, 
                            meta.classes
                        )
                except Exception as e:
                    OpLog(f"Visualization failed: {e}", bLines=False)
            


def Test(meta,ssd,model_file,transform):
    # model_file = r"D:\01.project\antig\mission7\data\pet_data\modelfiles\checkpoint_SSD300VGG16Transfer_partial_30_0.001_epoch_08.pth"
    # model_file = r"D:\01.project\antig\mission7\data\pet_data\modelfiles\checkpoints_SSD300_partial_20_0.001\SSD_epoch_18.pth"
    # model_file = r"D:\01.project\antig\mission7\data\pet_data\modelfiles\checkpoint_SSDLiteMobileNetV3Transfer_partial_30_0.001_epoch_11.pth"
    # 1. Instantiate the model class
    
    # 2. Load checkpoint state_dict
    state_dict = LoadMode(model_file, meta.device)
    
    # 3. Load state into model (strict=False allows partial/mismatched keys)
    try:
        result = ssd._model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            OpLog(f"Missing keys (will use random init): {len(result.missing_keys)} keys")
        if result.unexpected_keys:
            OpLog(f"Unexpected keys (will ignore): {len(result.unexpected_keys)} keys")
        OpLog(f"Model loaded from {model_file}")
    except Exception as e:
        OpLog(f"Failed to load model state: {e}")
        return
    
    # 4. Get transform from model
    
    # 5. Split train/val
    train_list, val_list = GetTrainValidationSplit(meta.df_trainval, test_size=0.3, random_state=42)
    test_list = meta.test_list
    
    # 6. Create test loader
    _, _, test_loader = GetLoader(
        meta=meta,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
        transform=transform,
        batchSize=8)

    # 7. Run test
    testModel(meta, ssd._model, test_loader, max_images=10)

def Test_YOLOv8n(meta, yolo_wrapper, model_file, transform):
    """
    Test function specifically for YOLOv8n models.
    
    Args:
        meta: MyMeta instance
        yolo_wrapper: YOLOv8nTransfer instance
        model_file: Path to checkpoint file (.pt)
        transform: Transform to apply to images
    """
    Lines(f"YOLOv8nTransfer 테스트를 시작합니다. (Checkpoint: {model_file})")
    
    # 1. Load checkpoint using YOLOv8nTransfer's method
    success = yolo_wrapper.load_checkpoint(model_file)
    if not success:
        OpLog(f"Failed to load YOLO checkpoint: {model_file}")
        return
    
    # 2. Split train/val
    train_list, val_list = GetTrainValidationSplit(meta.df_trainval, test_size=0.3, random_state=42)
    test_list = meta.test_list
    
    # 3. Create test loader
    _, _, test_loader = GetLoader(
        meta=meta,
        train_list=train_list,
        val_list=val_list,
        test_list=test_list,
        transform=transform,
        batchSize=8)
    
    # 4. Run YOLO-specific test
    device = meta.device
    yolo_wrapper._model.to(device)
    yolo_wrapper._model.eval()
    
    Lines(f"Train/Validation :{len(train_list)},{len(val_list)}")
    
    try:
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Test")):
                # YOLO: Process images one by one to avoid size mismatch
                for img_idx, img in enumerate(images):
                    img_batch = img.unsqueeze(0).to(device)
                    predictions = yolo_wrapper._model(img_batch)
                    
                    if batch_idx == 0 and img_idx < 3:
                        pred = predictions[0] if isinstance(predictions, list) else predictions
                        
                        if not isinstance(pred, dict):
                            try:
                                if hasattr(pred, 'boxes'):
                                    boxes = pred.boxes.xyxy.cpu() if hasattr(pred.boxes, 'xyxy') else torch.zeros((0, 4))
                                    labels = pred.boxes.cls.cpu().long() if hasattr(pred.boxes, 'cls') else torch.zeros((0,), dtype=torch.int64)
                                    scores = pred.boxes.conf.cpu() if hasattr(pred.boxes, 'conf') else torch.zeros((0,))
                                else:
                                    boxes = torch.zeros((0, 4))
                                    labels = torch.zeros((0,), dtype=torch.int64)
                                    scores = torch.zeros((0,))
                                pred = {'boxes': boxes, 'labels': labels, 'scores': scores}
                            except:
                                pred = {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,), dtype=torch.int64), 'scores': torch.zeros((0,))}
                        
                        visualize_prediction(img.cpu(), pred, meta.classes)
                
                if batch_idx >= 2:
                    break
        OpLog("YOLOv8n 테스트 완료")
        
    except Exception as e:
        OpLog(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def Test_fn():
    meta = MyMeta()
    ssd = SSD300VGG16Transfer(meta=meta)
    transform = ssd.get_default_transforms()
    model_file =r"D:\01.project\CodeIt\mission7\data\weight\checkpoint_SSD300VGG16Transfer_None_10_0.001_epoch_10.pth"
    Test(meta,ssd,model_file,transform)

    ssd = SSDLiteMobileNetV3Transfer(meta=meta)
    transform = ssd.get_default_transforms()
    model_file = r"D:\01.project\antig\mission7\data\pet_data\modelfiles\checkpoint_SSDLiteMobileNetV3Transfer_partial_30_0.001_epoch_11.pth"
    Test(meta,ssd,model_file,transform)


Lines("Define predict function.")

def Test_YOLO_fn():
    # Test YOLOv8n model with checkpoint
    meta = MyMeta()
    yolo = YOLOv8nTransfer(meta=meta)
    transform = yolo.get_default_transforms()
    model_file = r"D:\yolobest.pt"
    Test_YOLOv8n(meta, yolo, model_file, transform)
Test_YOLO_fn()