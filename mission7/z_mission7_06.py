import os
import random
import time
import datetime
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from tqdm.auto import tqdm
from collections import defaultdict
import os
import pandas as pd
import torch
import sys # <--- ADDED: sys 모듈 임포트 (OpLog 사용을 위해)
# python /content/drive/MyDrive/codeit/mission7/src/z_mission7_06.py
#════════════════════════════════════════
# ▣ 환경 설정. 
#════════════════════════════════════════
BASE_DIR = r"D:\05.gdrive\codeit\mission7\data\pet_data"
#DBASE_DIR = "/content/drive/MyDrive/codeit/mission7/data/pet_data"

## 메타 정보 클래스
class MyMeta():
    def __init__(self):
        # 1. 내부 변수는 앞에 _(언더스코어)를 붙여서 '내부용'임을 표시합니다.
        self._base_dir = BASE_DIR
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 경로 설정
        self._trainval_file = os.path.join(self._base_dir, "annotations", "annotations", "trainval.txt")
        self._test_file = os.path.join(self._base_dir, "annotations", "annotations", "test.txt")
        self._image_dir = os.path.join(self._base_dir, "images", "images")
        self._xml_dir = os.path.join(self._base_dir, "annotations", "annotations", "xmls")
        self._xml_files = [file for file in os.listdir(self._xml_dir) if file.endswith(".xml")]
        self._num_workers = 0  # 데이터 로더에 사용할 워커 수
        # 데이터프레임 로드
        try:
            self._df_trainval = pd.read_csv(self._trainval_file, sep="\s+", header=None)
            self._df_trainval.columns = ["Image", "ClassID", "Species", "BreedID"]
            
            self._df_test = pd.read_csv(self._test_file, sep="\s+", header=None)
            self._df_test.columns = ["Image", "ClassID", "Species", "BreedID"]
        except FileNotFoundError:
            print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
            self._df_trainval = None
            self._df_test = None
        self._ssd_weights_enum_available = False
        ## Try to import SSD weights enum support
        try:
            from torchvision.models.detection.ssd import SSD300_VGG16_Weights
            self._ssd_weights_enum_available = True
        except Exception:
            pass


    # === Properties (외부에서는 변수처럼 사용) ===
    @property
    def base_dir(self):
        return self._base_dir

    @property
    def device(self):
        return self._device

    @property
    def trainval_file(self):
        return self._trainval_file

    @property
    def test_file(self):
        return self._test_file
    
    @property
    def image_dir(self):
        return self._image_dir
    
    @property
    def xml_dir(self):
        return self._xml_dir

    @property
    def df_trainval(self):
        return self._df_trainval
    
    @property
    def df_test(self):
        return self._df_test

    @property
    def num_workers(self):
        return self._num_workers
    
    @property
    def ssd_weights_enum_available(self):
        return self._ssd_weights_enum_available
    @property
    def xml_files(self):
        return self._xml_files

#════════════════════════════════════════
# ▣ 유틸리티 함수 정의 
#════════════════════════════════════════
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

def OpLog(log,bLines=True):
    if bLines:
        Lines(f"[{now_str()}]{log}")
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

## IoU 계산 함수
## 두 박스의 좌표는 (xmin, ymin, xmax, ymax) 형식으로 전달
## 반환값: IoU 값 (0.0 ~ 1.0)
## 예: boxA = [50, 50, 150, 150], boxB = [100, 100, 200, 200]
## iou = iou_box(boxA, boxB)
## 두 박스가 겹치는 영역의 넓이를 두 박스의 합집합 넓이로 나눈 값
def iou_box(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - yb1)
    inter_area = inter_w * inter_h
    areaA = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    areaB = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0

Lines("Define Utility Function.")

## 테스트 함수.
def Test_Util_Functions():
    # Train/Test 메타 정보 읽기
    meta = MyMeta()
    df_trainval = meta.df_trainval
    df_test = meta.df_test
    Lines(f"TrainVal samples: {len(df_trainval)}")
    Lines(f"Test samples: {len(df_test)}")
    Lines(f"(XML Sapmles in xml_dir: {len(os.listdir(meta.xml_dir))})")
    # IoU 테스트
    boxA = [50, 50, 150, 150]
    boxB = [100, 100, 200, 200]
    iou = iou_box(boxA, boxB)
    Lines(f"IoU between boxA and boxB: {iou:.4f}")

Test_Util_Functions()


#════════════════════════════════════════
# ▣ 유틸리티 함수 및 클래스 정의 
#════════════════════════════════════════


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
    
    OpLog(f"Read meta: TrainVal={len(df_trainval)} Test={len(df_test)}", bLines=False) # <--- ADDED LOG

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
        OpLog(f"Detection dataset initialized. Original size={len(image_list)} Filtered size={len(self.image_list)}", bLines=False) # <--- ADDED LOG

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
    OpLog(f"Classification samples generated: Total={len(samples)} (BG_per_img={bg_per_image})", bLines=False) # <--- ADDED LOG
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
    def __init__(self, num_classes=3, gubun="partial", lr=1e-3, backbone_lr_ratio=0.1):
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
    def GetMyName(self) : return "EfficientNetB3"
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
    def __init__(self, num_classes=3, gubun="partial", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
            self.model = torchvision.models.swin_t(weights=weights)
        except Exception:
            self.model = torchvision.models.swin_t(pretrained=True)
        in_f = self.model.head.in_features
        self.model.head = nn.Linear(in_f, num_classes)
        self.gubun=gubun; self.lr=lr; self.backbone_lr_ratio=backbone_lr_ratio
    def GetMyName(self) : return "SwinTiny"
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
    def __init__(self, num_classes=3, gubun="partial", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        try:
            weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
            self.model = torchvision.models.mobilenet_v2(weights=weights)
        except Exception:
            self.model = torchvision.models.mobilenet_v2(pretrained=True)
        in_f = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_f, num_classes)
        self.gubun=gubun; self.lr=lr; self.backbone_lr_ratio=backbone_lr_ratio
    def GetMyName(self) : return "MobileNetV2"
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
    def __init__(self, num_classes=3, gubun="partial", lr=1e-3, backbone_lr_ratio=0.1):
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
    def GetMyName(self) : return "MobileNetV3"
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
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
class SSD300VGG16Transfer(nn.Module):
    def __init__(self, num_classes=3, gubun="partial", lr=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gubun = gubun
        self.lr = lr
        self.backbone_lr_ratio = backbone_lr_ratio
        meta = MyMeta() 

        # load ssd300_vgg16
        try:
            if  meta.ssd_weights_enum_available:
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
            # (num_anchors * num_classes) should include background (class 0)
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
    def GetMyName(self) : return "SSD300"
    def forward(self, images, targets=None):
        return self.model(images, targets)

    def setup_training(self):
        backbone_params=[]; head_params=[]
        for name,param in self.model.named_parameters():
            # head components contain classification_head and regression_head
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
        OpLog(f"Model built: {model_name} (Mode: det, Classes: {num_classes}, Gubun: {gubun})", bLines=False) # <--- ADDED LOG
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
        else:
            raise ValueError("Unknown classification model: " + model_name)
        opt = m.setup_training()
        OpLog(f"Model built: {model_name} (Mode: cls, Classes: {num_classes}, Gubun: {gubun})", bLines=False) # <--- ADDED LOG
        return m, opt

# -------------------------
# Trainer (branch based on mode)
# -------------------------
def calculate_mAP(all_preds, all_gts, iou_threshold=0.5):
    """
    Calculate mAP (Mean Average Precision) for a list of predictions and ground truths.
    This is a simplified implementation focusing on single-class mAP, adapted for multi-class.
    all_preds: list of dicts [{'boxes': tensor, 'labels': tensor, 'scores': tensor}]
    all_gts: list of dicts [{'boxes': tensor, 'labels': tensor}]
    Returns: mAP@iou_threshold
    """
    if not all_preds or not all_gts: return 0.0

    # Class 0 is background (ignored), 1=cat, 2=dog. Total 3 classes, but only 2 foreground classes.
    num_classes = 3 # includes background

    # Structure: {class_id: [{'iou': float, 'score': float, 'matched': bool, 'is_gt': bool}]}
    detections_by_class = defaultdict(list)
    gt_counts_by_class = defaultdict(int)
    
    # 1. Collect all predictions and ground truths, and establish matches
    for preds, gts in zip(all_preds, all_gts):
        gt_boxes = gts['boxes'].cpu().numpy()
        gt_labels = gts['labels'].cpu().numpy()
        pred_boxes = preds['boxes'].cpu().numpy()
        pred_labels = preds['labels'].cpu().numpy()
        pred_scores = preds['scores'].cpu().numpy()
        
        # 1.1. Count GTs
        for label in gt_labels:
            if label > 0: # ignore background (class 0)
                gt_counts_by_class[label] += 1

        # 1.2. Match predictions to GTs (greedy approach)
        if len(pred_boxes) == 0: continue

        # Sort predictions by score (descending)
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_labels = pred_labels[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        # Keep track of matched GTs
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        for i in range(len(pred_boxes)):
            p_box = pred_boxes[i]
            p_label = pred_labels[i]
            p_score = pred_scores[i]
            
            # Skip background predictions
            if p_label == 0: continue

            # Find best GT match for this prediction
            best_iou = 0.0
            best_gt_idx = -1
            
            for j in range(len(gt_boxes)):
                # Must be same class AND not already matched
                if gt_labels[j] == p_label and not gt_matched[j]:
                    current_iou = iou_box(p_box, gt_boxes[j])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt_idx = j
            
            # 1.3. Record True Positive (TP) or False Positive (FP)
            if best_gt_idx != -1 and best_iou >= iou_threshold:
                # True Positive
                detections_by_class[p_label].append({'score': p_score, 'tp': 1, 'fp': 0})
                gt_matched[best_gt_idx] = True # Mark GT as matched
            else:
                # False Positive
                detections_by_class[p_label].append({'score': p_score, 'tp': 0, 'fp': 1})

    # 2. Calculate Average Precision (AP) for each class
    aps = []
    for class_id in range(1, num_classes): # Iterate over foreground classes (1=cat, 2=dog)
        if class_id not in detections_by_class:
            if gt_counts_by_class[class_id] > 0:
                aps.append(0.0) # No detections but has GTs
            continue

        dets = detections_by_class[class_id]
        dets.sort(key=lambda x: x['score'], reverse=True) # Sort by score

        Tps = np.cumsum([d['tp'] for d in dets])
        Fps = np.cumsum([d['fp'] for d in dets])
        num_gts = gt_counts_by_class[class_id]

        if num_gts == 0:
            ap = 0.0 # No GTs for this class
        else:
            recalls = Tps / num_gts
            precisions = Tps / (Tps + Fps)

            # 11-point interpolation (standard COCO/PASCAL VOC 2007)
            # Find the max precision for recall >= r_i
            recall_thresholds = np.linspace(0., 1.0, 11)
            ap = 0.0
            for r_t in recall_thresholds:
                p_max = 0.0
                if np.any(recalls >= r_t):
                    p_max = np.max(precisions[recalls >= r_t])
                ap += p_max / 11.0
        
        aps.append(ap)
        # print(f"Class {class_id} AP: {ap:.4f} (GTs: {num_gts})")

    # 3. Calculate mAP
    mean_ap = np.mean(aps) if len(aps) > 0 else 0.0
    return mean_ap

def calculate_cls_metrics(all_preds: np.ndarray, all_labels: np.ndarray, num_classes: int):
    """
    Calculate classification metrics (Confusion Matrix, Recall, Precision, F1-score).
    Metrics are calculated per class (excluding class 0: background) and macro-averaged.
    all_preds/all_labels: 1D numpy array of class labels (0, 1, 2, ...)
    """
    
    # 1. Build Confusion Matrix (CM)
    # CM[i, j] is the count of samples with true class i and predicted class j
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        if true < num_classes and pred < num_classes:
            cm[true, pred] += 1
        
    metrics = defaultdict(list)
    
    # 2. Calculate metrics for each foreground class (1, 2, ...)
    # Class 0 is background, we typically ignore it in multi-class macro-metrics.
    
    for class_id in range(1, num_classes):
        # True Positives: Class_id correctly predicted as Class_id
        TP = cm[class_id, class_id]
        
        # False Positives: Other classes predicted as Class_id (sum of column for class_id - TP)
        FP = np.sum(cm[:, class_id]) - TP
        
        # False Negatives: Class_id predicted as other classes (sum of row for class_id - TP)
        FN = np.sum(cm[class_id, :]) - TP
        
        # Avoid division by zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1_score)
        
        # Include per-class metrics
        metrics[f'precision_cls_{class_id}'] = precision
        metrics[f'recall_cls_{class_id}'] = recall
        metrics[f'f1_score_cls_{class_id}'] = f1_score
        
    # 3. Macro-average the metrics
    macro_precision = np.mean(metrics['precision']) if metrics['precision'] else 0.0
    macro_recall = np.mean(metrics['recall']) if metrics['recall'] else 0.0
    macro_f1_score = np.mean(metrics['f1_score']) if metrics['f1_score'] else 0.0
    
    results = {
        'confusion_matrix': cm.tolist(), # Convert to list for easier saving
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1_score': macro_f1_score
    }
    
    # Add per-class metrics to results
    for k, v in metrics.items():
        if k.startswith('precision_cls_') or k.startswith('recall_cls_') or k.startswith('f1_score_cls_'):
            results[k] = v

    return results


def save_model_and_results(save_dir: str, mode: str, model_name: str, gubun: str, lr: float, epochs: int, test_results: Dict):
    """Save the best model state and append results to a CSV file."""
    if save_dir is None:
        print("Save directory not specified. Skipping model and results saving.")
        return
    
    makedirs(save_dir)
    results_path = os.path.join(save_dir, "results.csv")
    
    # 1. CSV 결과 기록
    result_row = {
        "timestamp": now_str(),
        "mode": mode,
        "model": model_name,
        "gubun": gubun,
        "lr": lr,
        "epochs": epochs,
    }
    result_row.update(test_results)
    
    df_results = pd.DataFrame([result_row])
    
    # Check if CSV exists to decide on writing header
    if not os.path.exists(results_path):
        df_results.to_csv(results_path, index=False)
    else:
        df_results.to_csv(results_path, mode='a', header=False, index=False)
    
    print(f"Results saved to {results_path}")


class TrainerUnified:
    def __init__(self, mode, model, optimizer, device, train_loader, val_loader, test_loader=None, scheduler=None, 
                 model_name="", gubun="", lr=0.0, epochs=0, num_classes=3): # <--- MODIFIED: num_classes 추가
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
        
        # for saving
        self.model_name = model_name
        self.gubun = gubun
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes # <--- ADDED

        self.model = self.model.to(device)

    def train_one_epoch_cls(self):
        self.model.train()
        running_loss = 0.0
        correct = 0; total = 0
        index = 0
        for imgs, labels in tqdm(self.train_loader, desc="Train cls", leave=False,disable=True):
            imgs = imgs.to(self.device); labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outs = self.model(imgs)
            loss = self.criterion(outs, labels)
            loss.backward(); self.optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            OpLog(f"★  {index}/{len(self.train_loader)}, Train Loss: {loss:.4f}", bLines=False)
            print(f"[{now_str()}]★  {index}/{len(self.train_loader)}, Train Loss: {loss:.4f}", end="\r")
            
            index += 1
        return running_loss / total, correct / total

    def eval_cls(self, loader, return_preds=False): # <--- MODIFIED: return_preds 인자 추가
        self.model.eval()
        running_loss = 0.0; correct = 0; total = 0
        all_preds = []; all_labels = [] # <--- ADDED
        with torch.no_grad():
            for imgs, labels in tqdm(loader, desc="Eval cls", leave=False):
                imgs = imgs.to(self.device); labels = labels.to(self.device)
                outs = self.model(imgs)
                loss = self.criterion(outs, labels)
                running_loss += loss.item() * imgs.size(0)
                preds = outs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                if return_preds: # <--- ADDED
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

        if return_preds: # <--- ADDED
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            # return_preds가 True일 때 (loss, acc, preds, labels) 반환
            return running_loss / total if total>0 else 0.0, correct / total if total>0 else 0.0, all_preds, all_labels
            
        return running_loss / total if total>0 else 0.0, correct / total if total>0 else 0.0

    def train_one_epoch_det(self):
        self.model.train()
        running_loss = 0.0
        index = 1 
        for images, targets in tqdm(self.train_loader, desc="Train det", leave=False,disable=True):
            images = [img.to(self.device) for img in images]
            t_targets = [{k: v.to(self.device) for k,v in t.items()} for t in targets]
            self.optimizer.zero_grad()
            loss_dict = self.model(images, t_targets)
            losses = sum(v for v in loss_dict.values())
            losses.backward()
            self.optimizer.step()
            running_loss += losses.item()
            OpLog(f"★  {index}/{len(self.train_loader)}, Train Loss: {losses:.4f}")
            print(f"[{now_str()}]★  {index}/{len(self.train_loader)}, Train Loss: {losses:.4f}", end="\r")
         
            index += 1
        return running_loss / len(self.train_loader)

    def eval_det_proper(self, loader):
        """Proper evaluation using mAP (Mean Average Precision)"""
        self.model.eval()
        all_preds = []
        all_gts = []
        with torch.no_grad():
            for images, targets in tqdm(loader, desc="Eval det (mAP)", leave=False):
                images = [img.to(self.device) for img in images]
                # targets are already on CPU/Numpy in dataloader, so move to CPU only for list append
                all_gts.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
                
                preds = self.model(images)
                all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])

        mAP_50 = calculate_mAP(all_preds, all_gts, iou_threshold=0.5)
        mAP_75 = calculate_mAP(all_preds, all_gts, iou_threshold=0.75)
        # Simplified: just mAP@0.5 and mAP@0.75
        return {"mAP@0.5": mAP_50, "mAP@0.75": mAP_75}

    def fit(self, epochs, save_dir=None):
        OpLog(f"Trainer starting fit. Mode: {self.mode}, Model: {self.model_name}, Epochs: {epochs}", bLines=False) # <--- ADDED LOG
        best_metric = -1.0
        best_epoch = -1
        
        # Placeholder for final test results (will be filled with best model's results)
        final_test_results = {} 

        for epoch in range(1, epochs+1):
            t0 = time.time()
            if self.mode == "cls":
                OpLog(f"Start Train - {self.model.GetMyName()}, epoch:{epoch}/{epochs}")
                tr_loss, tr_acc = self.train_one_epoch_cls()
                val_loss, val_acc = self.eval_cls(self.val_loader)
                
                OpLog(f"Epoch {epoch} cls train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={(time.time()-t0):.1f}s", bLines=False)
                # Check for best model (based on val_acc)
                if val_acc > best_metric:
                    best_metric = val_acc
                    best_epoch = epoch
                    if save_dir:
                        makedirs(save_dir)
                        # Save best model checkpoint
                        ckpt = {"epoch":epoch, "model_state":self.model.state_dict(), "optimizer_state":self.optimizer.state_dict(), "val_acc":val_acc}
                        torch.save(ckpt, os.path.join(save_dir, f"cls_best.pth"))
                        print(f"-> Saved best model at epoch {epoch} with val_acc={val_acc:.4f}")
                        
                if self.scheduler is not None:
                    self.scheduler.step()

            else: # Detection
                OpLog(f"Start Train - {self.model.GetMyName()}, epoch:{epoch}/{epochs}")
                tr_loss = self.train_one_epoch_det()
                val_metric = self.eval_det_proper(self.val_loader)
                val_mAP = val_metric['mAP@0.5'] # Use mAP@0.5 as the main metric
                OpLog(f"Epoch {epoch} det train_loss={tr_loss:.4f} val_mAP@0.5={val_mAP:.4f} val_mAP@0.75={val_metric['mAP@0.75']:.4f} time={(time.time()-t0):.1f}s", bLines=False)
                
                # Check for best model (based on mAP@0.5)
                if val_mAP > best_metric:
                    best_metric = val_mAP
                    best_epoch = epoch
                    if save_dir:
                        makedirs(save_dir)
                        # Save best model checkpoint
                        ckpt = {"epoch":epoch, "model_state":self.model.state_dict(), "optimizer_state":self.optimizer.state_dict(), "val_metric":val_metric}
                        torch.save(ckpt, os.path.join(save_dir, f"det_best.pth"))
                        print(f"-> Saved best model at epoch {epoch} with val_mAP@0.5={val_mAP:.4f}")
                        
                if self.scheduler is not None:
                    self.scheduler.step()

        # Final Eval using the best model
        print("==" + "═" * 20 + f" Final Eval (Best Epoch: {best_epoch}) " + "═" * 20 + "==")
        if save_dir and best_epoch != -1:
            # Load best model for final test
            if self.mode == "cls":
                best_ckpt_path = os.path.join(save_dir, "cls_best.pth")
            else:
                best_ckpt_path = os.path.join(save_dir, "det_best.pth")
                
            if os.path.exists(best_ckpt_path):
                print(f"Loading best model from {best_ckpt_path}...")
                ckpt = torch.load(best_ckpt_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state"])
            else:
                print("Warning: Best checkpoint not found. Using last epoch model for test.")

        if self.mode == "cls" and self.test_loader is not None:
            # <--- MODIFIED: eval_cls 호출 시 return_preds=True 추가
            test_loss, test_acc, all_preds, all_labels = self.eval_cls(self.test_loader, return_preds=True)
            print(f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")
            
            # <--- ADDED: Classification Metrics Calculation
            cls_metrics = calculate_cls_metrics(all_preds, all_labels, num_classes=self.num_classes)
            
            print(f"Test Macro-Avg: Precision={cls_metrics['macro_precision']:.4f}, Recall={cls_metrics['macro_recall']:.4f}, F1-Score={cls_metrics['macro_f1_score']:.4f}")
            print("Confusion Matrix (True\\Pred):")
            print(np.array(cls_metrics['confusion_matrix']))

            final_test_results = {
                "test_loss": test_loss, 
                "test_acc": test_acc, 
                "best_epoch": best_epoch,
                "confusion_matrix": str(cls_metrics['confusion_matrix']) # CSV 저장을 위해 문자열로 변환
            }
            # Macro-metrics 및 Class-별 metrics 추가 (혼동행렬은 이미 추가됨)
            final_test_results.update({k:v for k,v in cls_metrics.items() if k != 'confusion_matrix'})
            
        elif self.mode == "det" and self.test_loader is not None:
            m = self.eval_det_proper(self.test_loader)
            print("Test det metric (mAP@0.5):", m['mAP@0.5'], "(mAP@0.75):", m['mAP@0.75'])
            final_test_results = {"test_mAP_0.5": m['mAP@0.5'], "test_mAP_0.75": m['mAP@0.75'], "best_epoch": best_epoch}

        # Save results to CSV
        save_model_and_results(
            save_dir=save_dir, 
            mode=self.mode, 
            model_name=self.model_name, 
            gubun=self.gubun, 
            lr=self.lr, 
            epochs=self.epochs, 
            test_results=final_test_results
        )
        OpLog(f"Trainer finished fit for {self.model_name}. Best epoch: {best_epoch}", bLines=False) # <--- ADDED LOG


# -------------------------
# Execute_fn (unified entry)
# -------------------------
def Execute_fn(
    base_dir: str,
    model_name: str = "EfficientNetB3",
    gubun: str = "partial",
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
    model_name_lower = model_name.lower()
    if( model_name_lower == "ssd"):
        mode = "det"
    else:
        mode = "cls"
    
    OpLog(f"Starting execution for {mode} mode with model {model_name} (Epochs={epochs}, LR={lr}, Gubun={gubun})", bLines=True) # <--- ADDED LOG (START)


    # read meta
    df_trainval, df_test, image_dir, xml_dir = read_meta(base_dir)
    OpLog(f"Data meta read from {base_dir}", bLines=False) # <--- ADDED LOG

    device = DEVICE_TYPE

    # Use 3 classes: 0=background, 1=cat, 2=dog
    num_classes = 3

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
        
        OpLog(f"Detection training setup complete. Training on {len(train_ds)} samples.", bLines=False) # <--- ADDED LOG

        # model
        model, optimizer = make_model("det", "SSD", num_classes=num_classes, gubun=gubun, lr=lr, ratio=ratio)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        trainer = TrainerUnified(
            mode="det", model=model, optimizer=optimizer, device=device, 
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
            scheduler=scheduler, model_name="SSD", gubun=gubun, lr=lr, epochs=epochs,
            num_classes=num_classes # <--- ADDED
        )
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
        
        OpLog(f"Classification training setup complete. Training on {len(train_ds)} samples.", bLines=False) # <--- ADDED LOG

        # model
        model, optimizer = make_model("cls", model_name, num_classes=num_classes, gubun=gubun, lr=lr, ratio=ratio)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        trainer = TrainerUnified(
            mode="cls", model=model, optimizer=optimizer, device=device, 
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
            scheduler=scheduler, model_name=model_name, gubun=gubun, lr=lr, epochs=epochs,
            num_classes=num_classes # <--- ADDED
        )
        trainer.fit(epochs, save_dir=save_dir)

    OpLog("Execute_fn finished.", bLines=True) # <--- ADDED LOG (END)


# -------------------------
# Experiments
# -------------------------
def Execute_Multi01(gubun="partial", epochs=10, batch_size=32, lr=1e-3):
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

def Execute_Multi02(gubun="partial", epochs=10, batch_size=32, lr=1e-3):
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
    
Execute_Multi02(epochs=10, batch_size=16, lr=1e-3)
Execute_Multi02(epochs=30, batch_size=16, lr=1e-3)
Execute_Multi02(epochs=10, batch_size=16, lr= 0.0001)
Execute_Multi02(epochs=30, batch_size=16, lr= 0.0001)
Execute_Multi01(epochs=10, batch_size=16, lr=1e-3)
Execute_Multi01(epochs=30, batch_size=16, lr=1e-3)
Execute_Multi01(epochs=10, batch_size=16, lr= 0.0001)
Execute_Multi01(epochs=30, batch_size=16, lr= 0.0001)
