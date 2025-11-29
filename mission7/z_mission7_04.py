

import os
import random
import time
import datetime
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from tqdm.auto import tqdm

# Try to import SSD weights enum support
try:
    from torchvision.models.detection.ssd import SSD300_VGG16_Weights
    SSD_WEIGHTS_ENUM_AVAILABLE = True
except Exception:
    SSD_WEIGHTS_ENUM_AVAILABLE = False

NUM_WORKERS = 0
BASE_DIR = r"D:\05.gdrive\codeit\mission7\data\pet_data"
DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------
# Utilities
# -------------------------

## 라인 구분 함수
def Lines(text="", count=100):
    print("═" * count)
    if text != "":
        print(f"{text}")
        print("═" * count)


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def makedirs(d):
    os.makedirs(d, exist_ok=True)

def iou_box(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    areaB = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0

def OpLog(log,bLines=True):
    if bLines:
        Lines(log)
    # 호출한 함수 이름 가져오기
    # sys._getframe(1)은 OpLog을 호출한 함수의 프레임. 
    try:
        caller_name = sys._getframe(1).f_code.co_name
    except Exception:
        caller_name = "UnknownFunction"
        
    # 3. 로그 파일명 및 내용 포맷팅
    log_filename = f"{BASE_DIR}/op_log.txt"
    log_content = f"[{now_str()}] {caller_name}: {log}\n"
    # 4. 파일에 로그 추가 (append)
    try:
        # 'a' 모드는 파일이 없으면 생성하고, 있으면 기존 내용에 추가.
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(log_content)
    except Exception as e:
        print(f"로그 파일 쓰기 오류 발생: {e}")
Lines("Define Utility.")

# -------------------------
# Read dataset meta (Kaggle Oxford-IIIT Pet)
# -------------------------
def read_meta(base_dir):
    base_dir = str(base_dir)
    trainval_file = os.path.join(base_dir, "annotations", "annotations", "trainval.txt")
    test_file = os.path.join(base_dir, "annotations", "annotations", "test.txt")
    image_dir = os.path.join(base_dir, "images", "images")
    xml_dir = os.path.join(base_dir, "annotations", "annotations", "xmls")

    df_trainval = pd.read_csv(trainval_file, sep="\s+", header=None)
    df_trainval.columns = ["Image","ClassID","Species","BreedID"]
    df_test = pd.read_csv(test_file, sep="\s+", header=None)
    df_test.columns = ["Image","ClassID","Species","BreedID"]

    return df_trainval, df_test, image_dir, xml_dir

# -------------------------
# Detection Dataset (for SSD) - returns (image_tensor, target_dict)
# note: transforms here must be applied to image AND boxes must be scaled accordingly.
# -------------------------
class PetDetectionDataset(Dataset):
    def __init__(self, image_list: List[str], image_dir: str, xml_dir: str, transform=None, target_size=300):
        """
        image_list: list of basenames (without extension) from trainval/test txt
        """
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.image_list = [os.path.splitext(x)[0] for x in image_list]
        self.transform = transform  # should include Resize to (target_size, target_size) OR None
        self.target_size = target_size
        # filter only those with xml and image
        filtered = []
        for base in self.image_list:
            if os.path.exists(os.path.join(self.xml_dir, base + ".xml")) and os.path.exists(os.path.join(self.image_dir, base + ".jpg")):
                filtered.append(base)
        self.image_list = filtered

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        base = self.image_list[idx]
        img_path = os.path.join(self.image_dir, base + ".jpg")
        xml_path = os.path.join(self.xml_dir, base + ".xml")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # parse xml
        boxes = []
        labels = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            # map to 1=cat,2=dog (we keep these labels; SSD expects labels starting from 1)
            if "cat" in name:
                labels.append(1)
            else:
                labels.append(2)

        boxes = np.array(boxes, dtype=np.float32) if len(boxes)>0 else np.zeros((0,4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if len(labels)>0 else np.zeros((0,), dtype=np.int64)

        # apply transforms: if transform is resize to (target_size,target_size), scale boxes accordingly
        if self.transform is not None:
            # We assume transform contains resize to target_size as first op OR is T.Compose with Resize
            # To be safe, we do a simple resize float scaling ourselves (resize keeps aspect -> use exact scaling)
            # Resize image to target_size x target_size
            img_resized = img.resize((self.target_size, self.target_size))
            sx = self.target_size / W
            sy = self.target_size / H
            if boxes.shape[0] > 0:
                boxes_scaled = boxes.copy()
                boxes_scaled[:, 0] = boxes[:, 0] * sx
                boxes_scaled[:, 1] = boxes[:, 1] * sy
                boxes_scaled[:, 2] = boxes[:, 2] * sx
                boxes_scaled[:, 3] = boxes[:, 3] * sy
            else:
                boxes_scaled = boxes
            # then apply any additional transforms (ToTensor, Normalize, etc.)
            if isinstance(self.transform, T.Compose):
                # build a temporary compose that expects PIL image only (we already handled resize)
                # find last part that includes ToTensor and Normalize — apply entire compose to resized image
                img_tensor = self.transform(img_resized)
            else:
                img_tensor = T.ToTensor()(img_resized)
            target = {
                "boxes": torch.tensor(boxes_scaled, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
            return img_tensor, target
        else:
            # fallback: return raw tensor resized
            img_resized = img.resize((self.target_size, self.target_size))
            img_tensor = T.ToTensor()(img_resized)
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
            return img_tensor, target

# collate fn for detection (list of tuples -> tuple of lists)
def detection_collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets

# -------------------------
# Build classification samples from XML bboxes + background generation
# -------------------------
def parse_xml_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    names = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        boxes.append([xmin, ymin, xmax, ymax])
        names.append(name)
    return names, boxes

def generate_bbox_samples_from_xml(df_trainval, df_test, image_dir, xml_dir, bg_per_image=1, seed=42):
    """
    Returns list of samples: each sample = {'img': path, 'bbox':[xmin,ymin,xmax,ymax], 'label': 0/1/2}
    label: 0=background,1=cat,2=dog
    """
    random.seed(seed); np.random.seed(seed)
    df_all = pd.concat([df_trainval, df_test], ignore_index=True)
    img2species = dict(zip(df_all['Image'].astype(str), df_all['Species'].astype(int)))  # 1=cat,2=dog

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    samples = []

    for xml_file in tqdm(xml_files, desc="parse xml"):
        base = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(xml_dir, xml_file)
        img_path = os.path.join(image_dir, base + ".jpg")
        if not os.path.exists(img_path):
            continue
        names, boxes = parse_xml_boxes(xml_path)
        # foreground per bounding box
        for name, box in zip(names, boxes):
            # determine label: prefer image-level mapping
            sp = img2species.get(base, None)
            if sp is not None:
                label = int(sp)  # 1 or 2
            else:
                label = 1 if 'cat' in name.lower() else 2
            samples.append({'img': img_path, 'bbox': box, 'label': label})
        # background crops
        try:
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
        except Exception:
            continue
        # target crop size similar to average bbox
        if len(boxes) > 0:
            widths = [b[2]-b[0] for b in boxes]
            heights = [b[3]-b[1] for b in boxes]
            cw = int(np.mean(widths)); ch = int(np.mean(heights))
            cw = max(32, min(cw, W//2)); ch = max(32, min(ch, H//2))
        else:
            cw = max(32, min(W//4, 256)); ch = max(32, min(H//4, 256))
        created = 0; attempts = 0
        max_attempts = 50 * bg_per_image
        while created < bg_per_image and attempts < max_attempts:
            attempts += 1
            x1 = random.randint(0, max(0, W - cw))
            y1 = random.randint(0, max(0, H - ch))
            x2 = x1 + cw; y2 = y1 + ch
            crop = [x1,y1,x2,y2]
            ok = True
            for b in boxes:
                if iou_box(crop, b) >= 0.1:
                    ok = False; break
            if ok:
                samples.append({'img': img_path, 'bbox': crop, 'label': 0})
                created += 1
    random.shuffle(samples)
    return samples

# -------------------------
# Classification dataset from samples (on-the-fly crop + transform)
# -------------------------
class BBoxCropDataset(Dataset):
    def __init__(self, samples, transform=None, out_size=224):
        """
        samples: list of dicts {'img':path, 'bbox':[xmin,ymin,xmax,ymax], 'label':0/1/2}
        """
        self.samples = samples
        self.transform = transform
        self.out_size = out_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['img']).convert("RGB")
        xmin,ymin,xmax,ymax = s['bbox']
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(img.width, xmax); ymax = min(img.height, ymax)
        if xmax <= xmin or ymax <= ymin:
            crop = img
        else:
            crop = img.crop((xmin, ymin, xmax, ymax))
        label = int(s['label'])
        if self.transform:
            crop = self.transform(crop)
        else:
            crop = T.Resize((self.out_size,self.out_size))(crop)
            crop = T.ToTensor()(crop)
        return crop, label

# -------------------------
# Transforms for classification / detection
# -------------------------
def get_classification_transforms(model_name, is_train=True, size=224):
    if model_name.lower() == "efficientnetb3":
        out = 300
    else:
        out = size
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(out, scale=(0.7,1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1,0.1,0.1),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize(int(out*1.14)),
            T.CenterCrop(out),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

def get_detection_transforms(is_train=True):
    # For detection we will resize to 300x300 ourselves in dataset; these transforms assume PIL input size 300x300
    if is_train:
        return T.Compose([
            T.ColorJitter(0.1,0.1,0.1),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

# -------------------------
# Model classes (transfer) - classification
# -------------------------
class BasicTransfer(nn.Module):
    def __init__(self): super().__init__()

    def setup_training(self):
        raise NotImplementedError

class EfficientNetB3Transfer(BasicTransfer):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1
            self.model = torchvision.models.efficientnet_b3(weights=weights)
        except Exception:
            self.model = torchvision.models.efficientnet_b3(pretrained=True)
        # replace classifier
        try:
            in_f = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_f, num_classes)
        except Exception:
            self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.model.classifier.in_features, num_classes))
        self.gubun=gubun; self.lr=lr; self.backbone_lr_ratio=backbone_lr_ratio

    def forward(self,x): return self.model(x)

    def setup_training(self):
        head_params=[]; backbone_params=[]
        for name,p in self.model.named_parameters():
            if "classifier" in name or "fc" in name:
                p.requires_grad=True; head_params.append(p)
            else:
                if self.gubun=="freeze": p.requires_grad=False
                else: p.requires_grad=True; backbone_params.append(p)
        groups=[{'params':head_params,'lr':self.lr}]
        if backbone_params: groups.append({'params':backbone_params,'lr':self.lr*self.backbone_lr_ratio})
        return torch.optim.Adam(groups, lr=self.lr)

class SwinTinyTransfer(BasicTransfer):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_t(weights=weights)
        except Exception:
            self.model = torchvision.models.swin_t(pretrained=True)
        in_f = self.model.head.in_features
        self.model.head = nn.Linear(in_f, num_classes)
        self.gubun=gubun; self.lr=lr; self.backbone_lr_ratio=backbone_lr_ratio

    def forward(self,x): return self.model(x)

    def setup_training(self):
        head_params=[]; backbone_params=[]
        for name,p in self.model.named_parameters():
            if "head" in name:
                p.requires_grad=True; head_params.append(p)
            else:
                if self.gubun=="freeze": p.requires_grad=False
                else: p.requires_grad=True; backbone_params.append(p)
        groups=[{'params':head_params,'lr':self.lr}]
        if backbone_params: groups.append({'params':backbone_params,'lr':self.lr*self.backbone_lr_ratio})
        return torch.optim.Adam(groups, lr=self.lr)

class MobileNetV2Transfer(BasicTransfer):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
            self.model = torchvision.models.mobilenet_v2(weights=weights)
        except Exception:
            self.model = torchvision.models.mobilenet_v2(pretrained=True)
        in_f = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_f, num_classes)
        self.gubun=gubun; self.lr=lr; self.backbone_lr_ratio=backbone_lr_ratio

    def forward(self,x): return self.model(x)

    def setup_training(self):
        head_params=[]; backbone_params=[]
        for name,p in self.model.named_parameters():
            if "classifier" in name:
                p.requires_grad=True; head_params.append(p)
            else:
                if self.gubun=="freeze": p.requires_grad=False
                else: p.requires_grad=True; backbone_params.append(p)
        groups=[{'params':head_params,'lr':self.lr}]
        if backbone_params: groups.append({'params':backbone_params,'lr':self.lr*self.backbone_lr_ratio})
        return torch.optim.Adam(groups, lr=self.lr)

class MobileNetV3Transfer(BasicTransfer):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.model = torchvision.models.mobilenet_v3_large(weights=weights)
        except Exception:
            self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        # replace last classifier layer
        try:
            in_f = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_f, num_classes)
        except Exception:
            # fallback
            last = list(self.model.classifier.children())[-1]
            in_f = last.in_features
            self.model.classifier[-1] = nn.Linear(in_f, num_classes)
        self.gubun=gubun; self.lr=lr; self.backbone_lr_ratio=backbone_lr_ratio

    def forward(self,x): return self.model(x)

    def setup_training(self):
        head_params=[]; backbone_params=[]
        for name,p in self.model.named_parameters():
            if "classifier" in name:
                p.requires_grad=True; head_params.append(p)
            else:
                if self.gubun=="freeze": p.requires_grad=False
                else: p.requires_grad=True; backbone_params.append(p)
        groups=[{'params':head_params,'lr':self.lr}]
        if backbone_params: groups.append({'params':backbone_params,'lr':self.lr*self.backbone_lr_ratio})
        return torch.optim.Adam(groups, lr=self.lr)

# -------------------------
# SSD Transfer class (detection) - similar structure to classification transfer classes
# -------------------------
class SSD300VGG16Transfer(nn.Module):
    def __init__(self, num_classes=3, gubun="freeze", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gubun = gubun
        self.lr = lr
        self.backbone_lr_ratio = backbone_lr_ratio

        # load ssd300_vgg16
        try:
            if SSD_WEIGHTS_ENUM_AVAILABLE:
                weights = SSD300_VGG16_Weights.DEFAULT
                self.model = torchvision.models.detection.ssd300_vgg16(weights=weights)
            else:
                # older installed torchvision
                self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
        except Exception:
            # if weights unavailable, create model without weights
            self.model = torchvision.models.detection.ssd300_vgg16(weights=None)

        # Modify classification head to output num_classes
        # torchvision's head has attribute classification_head with conv and num_classes fields
        try:
            # get in_channels and anchors
            ch = self.model.head.classification_head
            in_ch = ch.conv[0].in_channels if isinstance(ch.conv, nn.Sequential) else ch.conv.in_channels
            num_anchors = ch.num_anchors
            new_conv = nn.Conv2d(in_ch, num_anchors * num_classes, kernel_size=3, padding=1)
            # replace conv - keep as Sequential for safety
            self.model.head.classification_head.conv = nn.Sequential(new_conv)
            # update param if exists
            self.model.head.classification_head.num_classes = num_classes
        except Exception:
            # If structure differs, attempt a safer replacement by setting attr if present
            try:
                self.model.head.classification_head.num_classes = num_classes
            except Exception:
                pass

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def setup_training(self):
        backbone_params=[]; head_params=[]
        for name,param in self.model.named_parameters():
            if "head.classification_head" in name or "head.regression_head" in name or "head" in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                if self.gubun == "freeze":
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    backbone_params.append(param)
        groups = [{'params': head_params, 'lr': self.lr}]
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': self.lr * self.backbone_lr_ratio})
        optimizer = torch.optim.SGD(groups, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        return optimizer

# -------------------------
# Model factory
# -------------------------
def make_model(mode: str, model_name: str, num_classes: int, gubun: str, lr: float, ratio: float):
    """
    mode: 'det' or 'cls'
    model_name: e.g. 'SSD' for detection, or 'EfficientNetB3','SwinTiny','MobileNetV2','MobileNetV3'
    """
    model_name = model_name.lower()
    if mode == "det":
        # SSD only option here
        model = SSD300VGG16Transfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
        optimizer = model.setup_training()
        return model, optimizer
    else:
        # classification
        if model_name == "efficientnetb3":
            m = EfficientNetB3Transfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
        elif model_name == "swintiny":
            m = SwinTinyTransfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
        elif model_name == "mobilenetv2":
            m = MobileNetV2Transfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
        elif model_name == "mobilenetv3":
            m = MobileNetV3Transfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
        elif model_name == "ssd":
            m = SSD300VGG16Transfer(num_classes=num_classes, gubun=gubun, lr=lr, backbone_lr_ratio=ratio)
        else:
            raise ValueError("Unknown classification model: " + model_name)
        opt = m.setup_training()
        return m, opt

# -------------------------
# Trainer (branch based on mode)
# -------------------------
class TrainerUnified:
    def __init__(self, mode, model, optimizer, device, train_loader, val_loader, test_loader=None, scheduler=None):
        """
        mode: 'det' or 'cls'
        model: model instance (for det: SSD300VGG16Transfer, for cls: BasicTransfer)
        optimizer: optimizer from model.setup_training()
        """
        self.mode = mode
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.criterion = nn.CrossEntropyLoss() if mode == "cls" else None

        self.model = self.model.to(device)

    def train_one_epoch_cls(self):
        self.model.train()
        running_loss = 0.0
        correct = 0; total = 0
        for imgs, labels in tqdm(self.train_loader, desc="Train cls", leave=False):
            imgs = imgs.to(self.device); labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outs = self.model(imgs)
            loss = self.criterion(outs, labels)
            loss.backward(); self.optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return running_loss / total, correct / total

    def eval_cls(self, loader):
        self.model.eval()
        running_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Eval cls", leave=False):
                imgs = imgs.to(self.device); labels = labels.to(self.device)
                outs = self.model(imgs)
                loss = self.criterion(outs, labels)
                running_loss += loss.item() * imgs.size(0)
                preds = outs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return running_loss / total if total>0 else 0.0, correct / total if total>0 else 0.0

    def train_one_epoch_det(self):
        self.model.train()
        running_loss = 0.0
        for images, targets in tqdm(self.train_loader, desc="Train det", leave=False):
            images = [img.to(self.device) for img in images]
            t_targets = [{k: v.to(self.device) for k,v in t.items()} for t in targets]
            self.optimizer.zero_grad()
            loss_dict = self.model(images, t_targets)
            losses = sum(v for v in loss_dict.values())
            losses.backward()
            self.optimizer.step()
            running_loss += losses.item()
        return running_loss / len(self.train_loader)

    def eval_det_simple(self, loader):
        # simplified eval: average predicted score; proper mAP requires full matching
        self.model.eval()
        scores = []
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Eval det", leave=False):
                images = [img.to(self.device) for img in images]
                preds = self.model(images)
                for p in preds:
                    if 'scores' in p and len(p['scores'])>0:
                        scores.append(p['scores'].cpu().numpy().mean())
        avg_score = float(np.mean(scores)) if len(scores)>0 else 0.0
        return {"avg_score": avg_score}

    def fit(self, epochs, save_dir=None):
        best_metric = -1
        for epoch in range(1, epochs+1):
            t0 = time.time()
            if self.mode == "cls":
                tr_loss, tr_acc = self.train_one_epoch_cls()
                val_loss, val_acc = self.eval_cls(self.val_loader)
                if self.scheduler is not None:
                    self.scheduler.step()
                print(f"[{now_str()}] Epoch {epoch} cls train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={(time.time()-t0):.1f}s")
                # save
                if save_dir:
                    makedirs(save_dir)
                    ckpt = {"epoch":epoch, "model_state":self.model.state_dict(), "optimizer_state":self.optimizer.state_dict(), "val_acc":val_acc}
                    torch.save(ckpt, os.path.join(save_dir, f"cls_epoch_{epoch}.pth"))
                if val_acc > best_metric:
                    best_metric = val_acc
            else:
                tr_loss = self.train_one_epoch_det()
                val_metric = self.eval_det_simple(self.val_loader)
                if self.scheduler is not None:
                    self.scheduler.step()
                print(f"[{now_str()}] Epoch {epoch} det train_loss={tr_loss:.4f} val_avg_score={val_metric['avg_score']:.4f} time={(time.time()-t0):.1f}s")
                if save_dir:
                    makedirs(save_dir)
                    ckpt = {"epoch":epoch, "model_state":self.model.state_dict(), "optimizer_state":self.optimizer.state_dict(), "val_metric":val_metric}
                    torch.save(ckpt, os.path.join(save_dir, f"det_epoch_{epoch}.pth"))
        # final test
        print("== Final Eval ==")
        if self.mode == "cls" and self.test_loader is not None:
            test_loss, test_acc = self.eval_cls(self.test_loader)
            print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")
        elif self.mode == "det" and self.test_loader is not None:
            m = self.eval_det_simple(self.test_loader)
            print("Test det metric (avg_score):", m)

# -------------------------
# Execute_fn (unified entry)
# -------------------------
def Execute_fn(
    base_dir: str,
    model_name: str = "EfficientNetB3",
    gubun: str = "freeze",
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    ratio: float = 0.1,
    bg_per_image: int = 1,
    val_ratio: float = 0.15,
    seed: int = 42,
    size: int = 224,
    step_size: int = 3,
    gamma: float = 0.1,
    save_dir: str = None
    ):
    """
    mode: 'cls' (classification from bbox crops) or 'det' (train SSD)
    model_name: for cls: EfficientNetB3,SwinTiny,MobileNetV2,MobileNetV3 ; for det: 'SSD'
    """
    model_name = model_name.lower()
    if( model_name == "ssd"):
        mode = "det"
    else:
        mode = "cls"

    # read meta
    df_trainval, df_test, image_dir, xml_dir = read_meta(base_dir)

    device = DEVICE_TYPE

    if mode == "det":
        # detection path
        # build lists from trainval/test files
        trainval_list = df_trainval['Image'].tolist()
        test_list = df_test['Image'].tolist()
        from sklearn.model_selection import train_test_split
        train_list, val_list = train_test_split(trainval_list, test_size=val_ratio, random_state=seed)
        # detection transforms
        det_train_tf = get_detection_transforms(is_train=True)
        det_val_tf = get_detection_transforms(is_train=False)
        train_ds = PetDetectionDataset(train_list, image_dir, xml_dir, transform=det_train_tf, target_size=300)
        val_ds   = PetDetectionDataset(val_list, image_dir, xml_dir, transform=det_val_tf, target_size=300)
        test_ds  = PetDetectionDataset(test_list, image_dir, xml_dir, transform=det_val_tf, target_size=300)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn, num_workers=NUM_WORKERS)
        # model
        model, optimizer = make_model("det", "SSD", num_classes=3, gubun=gubun, lr=lr, ratio=ratio)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        trainer = TrainerUnified(mode="det", model=model, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, scheduler=scheduler)
        trainer.fit(epochs, save_dir=save_dir)
    else:
        # classification path
        # Build samples from xml bboxes + background
        samples = generate_bbox_samples_from_xml(df_trainval, df_test, image_dir, xml_dir, bg_per_image=bg_per_image, seed=seed)
        if len(samples) == 0:
            print("No samples generated. Check xml_dir and image_dir paths.")
            return
        # split samples into train/val/test (stratified by label)
        df_samples = pd.DataFrame(samples)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df_samples, test_size=val_ratio, stratify=df_samples['label'], random_state=seed)
        # For test set use samples originating from df_test images only if available, else use val_df
        test_samples = [s for s in samples if os.path.splitext(os.path.basename(s['img']))[0] in set(df_test['Image'].astype(str).tolist())]
        if len(test_samples) > 0:
            test_df = pd.DataFrame(test_samples)
        else:
            test_df = val_df.copy()
        # transforms
        train_tf = get_classification_transforms(model_name, is_train=True, size=size)
        val_tf = get_classification_transforms(model_name, is_train=False, size=size)
        # datasets
        train_ds = BBoxCropDataset(train_df.to_dict('records'), transform=train_tf, out_size=size)
        val_ds   = BBoxCropDataset(val_df.to_dict('records'), transform=val_tf, out_size=size)
        test_ds  = BBoxCropDataset(test_df.to_dict('records'), transform=val_tf, out_size=size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print("Dataset sizes:", len(train_ds), len(val_ds), len(test_ds))
        # model
        model, optimizer = make_model("cls", model_name, num_classes=3, gubun=gubun, lr=lr, ratio=ratio)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        trainer = TrainerUnified(mode="cls", model=model, optimizer=optimizer, device=device, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, scheduler=scheduler)
        trainer.fit(epochs, save_dir=save_dir)

    print("Execute_fn finished.")

# -------------------------
# CLI main
# -------------------------
def main_cli():
    parser = argparse.ArgumentParser(description="Pet full pipeline (detection/classification) using Oxford-IIIT Pet dataset")
    parser.add_argument("--base_dir", type=str, required=True, help="path to pet_data root (images/, annotations/...)")
    parser.add_argument("--mode", type=str, default="cls", choices=["cls","det"], help="cls = classification (3-class bbox crops), det = SSD detection")
    parser.add_argument("--model", type=str, default="EfficientNetB3", choices=["EfficientNetB3","SwinTiny","MobileNetV2","MobileNetV3"], help="classification model (used when --mode cls)")
    parser.add_argument("--gubun", type=str, default="freeze", choices=["freeze","partial","full"], help="freeze/partial/full training strategy")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ratio", type=float, default=0.1, help="backbone lr ratio for partial")
    parser.add_argument("--bg_per_image", type=int, default=1, help="background crops per image for cls")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--step_size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    Execute_fn(
        base_dir=args.base_dir,
        mode="det" if args.mode=="det" else "cls",
        model_name=args.model,
        gubun=args.gubun,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ratio=args.ratio,
        bg_per_image=args.bg_per_image,
        val_ratio=args.val_ratio,
        seed=args.seed,
        size=args.size,
        step_size=args.step_size,
        gamma=args.gamma,
        save_dir=args.save_dir
    )
# -------------------------
# Experiments
# -------------------------
def Execute_Multi(gubun="partial", epochs=10, batch_size=32, lr=1e-3):
    Execute_fn(
        base_dir=BASE_DIR,
        model_name="SSD",
        gubun= gubun,
        epochs=epochs,
        batch_size=batch_size,
        lr = lr
    )
    Execute_fn(
        base_dir= BASE_DIR,
        model_name="EfficientNetB3",
        gubun= gubun,
        epochs=epochs,
        batch_size=batch_size,
        lr = lr
    )
    Execute_fn(
        base_dir=BASE_DIR,
        model_name="swintiny",
        gubun= gubun,
        epochs=epochs,
        batch_size=batch_size,
        lr = lr
    )
    Execute_fn(
        base_dir=BASE_DIR,
        model_name="mobilenetv2",
        gubun= gubun,
        epochs=epochs,
        batch_size=batch_size,
        lr = lr
    )
    Execute_fn(
        base_dir=BASE_DIR,
        model_name="mobilenetv3",
        gubun= gubun,
        epochs=epochs,
        batch_size=batch_size,
        lr = lr
    )
    
Execute_Multi(epochs=10, batch_size=32, lr=1e-3)
Execute_Multi(epochs=30, batch_size=32, lr=1e-3)
Execute_Multi(epochs=10, batch_size=32, lr= 0.0001)
Execute_Multi(epochs=30, batch_size=32, lr= 0.0001)



