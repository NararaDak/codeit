import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import (
    ssd300_vgg16,
    SSD300_VGG16_Weights,
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNetV3_Large_Weights
)
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import random

# ====================================================================
# 1. 환경 설정 및 유틸리티 함수 (PLACEHOLDERS)
# ====================================================================

# 사용자 환경에 맞게 데이터 경로를 반드시 수정하세요.
BASE_DIR = "D:\\05.gdrive\\codeit\\mission7\\data\\pet_data" 
DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Lines(msg, bLines=True):
    """유틸리티 함수: 로그 출력"""
    if bLines: print("=" * 50)
    print(msg)

def GetTrainValidationSplit(df, test_size=0.3, random_state=42):
    """유틸리티 함수: 데이터 분할 (trainval.txt 기준)"""
    # 실제 프로젝트에서는 sklearn.model_selection.train_test_split 사용 권장
    image_list = df['Image'].tolist()
    random.seed(random_state)
    random.shuffle(image_list)
    split_idx = int(len(image_list) * (1 - test_size))
    return image_list[:split_idx], image_list[split_idx:]

class BasicTransfer(nn.Module):
    """모든 전이 학습 모델의 기본 클래스"""
    def __init__(self):
        super().__init__()
    
    # 모든 모델 클래스에 이 속성이 정의되어야 함
    @property
    def get_default_transforms(self):
        raise NotImplementedError("Transforms must be defined in subclass")

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
                msg = f"[{self.getMyName()}/epohs:{self._epochs}/lr:{self._lr}/] {index}/{len(train_loader)} - Loss: {losses.item():.4f}"
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

class TestDataset(Dataset):
    """테스트 데이터셋 Placeholder (XML 없이 이미지 로드)"""
    def __init__(self, image_dir, image_list, transforms=None):
        self.image_dir = image_dir
        self.image_files = image_list
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx] + ".jpg"
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        target = {'boxes': torch.zeros((0, 4), dtype=torch.float32), 
                  'labels': torch.zeros((0,), dtype=torch.int64)}
                  
        if self.transforms:
            image, target = self.transforms(image, target) 
        else:
            from torchvision.transforms import ToTensor
            image = ToTensor()(image)

        return image, target


# ====================================================================
# 2. MyMeta 및 VOCDataset 클래스 (제공된 코드 기반)
# ====================================================================

class MyMeta():
    def __init__(self):
        # ... (생략: 기존 초기화 로직 유지) ...
        self._base_dir = BASE_DIR
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._trainval_file = os.path.join(self._base_dir, "annotations", "annotations", "trainval.txt")
        self._test_file = os.path.join(self._base_dir, "annotations", "annotations", "test.txt")
        self._image_dir = os.path.join(self._base_dir, "images", "images")
        self._xml_dir = os.path.join(self._base_dir, "annotations", "annotations", "xmls")
        self._num_workers = 0  
        self._classes = ["background", "dog", "cat"] # 미세 조정을 위한 클래스
        
        try:
            # os.listdir은 디렉토리가 존재해야 하므로, xml_files는 try-except 바깥에서 처리하는 것이 좋음
            self._xml_files = [file for file in os.listdir(self._xml_dir) if file.endswith(".xml")]
        except FileNotFoundError:
            self._xml_files = []

        try:
            self._df_trainval = pd.read_csv(self._trainval_file, sep="\s+", header=None)
            self._df_trainval.columns = ["Image", "ClassID", "Species", "BreedID"]
            
            self._df_test = pd.read_csv(self._test_file, sep="\s+", header=None)
            self._df_test.columns = ["Image", "ClassID", "Species", "BreedID"]

            self._trainval_list = self._df_trainval['Image'].tolist()
            self._test_list = self._df_test['Image'].tolist()

        except FileNotFoundError:
            Lines(f"파일을 찾을 수 없습니다. 경로를 확인해주세요: {self._base_dir}", bLines=False)
            self._df_trainval = None
            self._df_test = None
            self._trainval_list = []
            self._test_list = []
            
        self._ssd_weights_enum_available = False
        try:
            from torchvision.models.detection.ssd import SSD300_VGG16_Weights
            self._ssd_weights_enum_available = True
        except Exception:
            pass
            
    # Property 정의는 생략하고, 클래스 전체를 복사할 수 있도록 합니다.
    @property
    def device(self): return self._device
    @property
    def image_dir(self): return self._image_dir
    @property
    def xml_dir(self): return self._xml_dir
    @property
    def df_trainval(self): return self._df_trainval
    @property
    def num_workers(self): return self._num_workers
    @property
    def classes(self): return self._classes
    @property
    def test_list(self): return self._test_list
        
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
        except Exception:
            # GetLoader에서 필터링을 했으므로, 여기서 오류 발생은 데이터 불일치
            print(f"경고: 데이터셋 로딩 중 XML 오류 발생 ({annotation_path}). 건너뜀.")
            return torch.zeros((3, 1, 1), dtype=torch.float32), {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,))}

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.classes:
                continue
            labels.append(self.classes.index(class_name))

            bndbox = obj.find("bndbox")
            x_min = int(bndbox.find("xmin").text)
            y_min = int(bndbox.find("ymin").text)
            x_max = int(bndbox.find("xmax").text)
            y_max = int(bndbox.find("ymax").text)
            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {"boxes": boxes, "labels": labels}

        # Transform 적용 (모델별 최적화된 transforms 사용)
        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            from torchvision.transforms import ToTensor
            image = ToTensor()(image)
            
        return image, target


# ====================================================================
# 3. 모델 클래스 정의 (4가지)
# ====================================================================

# ----------------- 1. SSD300 VGG16 (원본 모델) -----------------
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
        
        self._weights = SSD300_VGG16_Weights.DEFAULT
        self._model = ssd300_vgg16(weights=self._weights).to(self._device)
        
        # Head 교체 로직
        in_channels = [512, 1024, 512, 256, 256, 256] 
        num_anchors_per_layer = self._model.anchor_generator.num_anchors_per_location()
        self._model.head.classification_head = SSDClassificationHead(in_channels, num_anchors_per_layer, self._num_classes).to(self._device)
        self._model.head.bbox_regression_head = SSDClassificationHead(in_channels, num_anchors_per_layer, 4).to(self._device)
        
    def getMyName(self): return "SSD300VGG16Transfer"
    
    @property
    def get_default_transforms(self):
        """Pre-trained Weights에 정의된 최적 Transform 반환"""
        return self._weights.transforms()

    def getOptimizer(self):
        # ... (Optimizer 로직 생략) ...
        if self._gubun == "partial":
            params = [{"params": self._model.backbone.parameters(), "lr": self._lr * self._backbone_lr_ratio},
                      {"params": self._model.head.parameters(), "lr": self._lr}]
        # ... (생략) ...
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr) # 간소화
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return optimizer, scheduler
    
    def train(self, train_loader, val_loader):
        optimizer, scheduler = self.getOptimizer()
        Lines(f"{self.getMyName()} Training ({self._gubun}) Start (LR: {self._lr}, Epochs: {self._epochs})")
        # 실제 학습 로직 생략
        for epoch in range(self._epochs): pass
        Lines(f"{self.getMyName()} Training End")


# ----------------- 2. Faster R-CNN ResNet50 (고정확도 Two-stage) -----------------
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
        
        self._weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self._model = fasterrcnn_resnet50_fpn(weights=self._weights).to(self._device)

        # Box Head의 Classifier 교체
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._num_classes).to(self._device)

    def getMyName(self): return "FasterRCNNResNet50Transfer"

    @property
    def get_default_transforms(self):
        return self._weights.transforms()
        
    def getOptimizer(self):
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr)
        return optimizer, None
        
    def train(self, train_loader, val_loader):
        Lines(f"{self.getMyName()} Training ({self._gubun}) Start (LR: {self._lr}, Epochs: {self._epochs})")
        for epoch in range(self._epochs): pass
        Lines(f"{self.getMyName()} Training End")


# ----------------- 3. RetinaNet ResNet50 (Focal Loss 기반 One-stage) -----------------
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
        
        self._weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        
        # num_classes만 지정하면 Head 자동 교체
        self._model = retinanet_resnet50_fpn(weights=self._weights, num_classes=self._num_classes).to(self._device)
        
    def getMyName(self): return "RetinaNetResNet50Transfer"

    @property
    def get_default_transforms(self):
        return self._weights.transforms()

    def getOptimizer(self):
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr)
        return optimizer, None
        
    def train(self, train_loader, val_loader):
        Lines(f"{self.getMyName()} Training ({self._gubun}) Start (LR: {self._lr}, Epochs: {self._epochs})")
        for epoch in range(self._epochs): pass
        Lines(f"{self.getMyName()} Training End")


# ----------------- 4. SSDLite MobileNetV3 (모바일 경량화 모델) -----------------
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
        
        self._weights = SSDLite320_MobileNetV3_Large_Weights.DEFAULT
        self._model = ssdlite320_mobilenet_v3_large(weights=self._weights).to(self._device)

        # Head 교체 로직
        in_channels = [ch.in_channels for ch in self._model.head.classification_head]
        num_anchors_per_layer = self._model.anchor_generator.num_anchors_per_location()

        self._model.head.classification_head = SSDClassificationHead(in_channels, num_anchors_per_layer, self._num_classes).to(self._device)
        self._model.head.bbox_regression_head = SSDClassificationHead(in_channels, num_anchors_per_layer, 4).to(self._device)
        
    def getMyName(self): return "SSDLiteMobileNetV3Transfer"

    @property
    def get_default_transforms(self):
        return self._weights.transforms()

    def getOptimizer(self):
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr)
        return optimizer, None

    def train(self, train_loader, val_loader):
        Lines(f"{self.getMyName()} Training ({self._gubun}) Start (LR: {self._lr}, Epochs: {self._epochs})")
        for epoch in range(self._epochs): pass
        Lines(f"{self.getMyName()} Training End")


# ====================================================================
# 4. GetLoader 함수 (XML 필터링 로직 포함)
# ====================================================================

def GetLoader(meta, train_list_all, val_list_all, test_list_all, transform, batchSize=8):
    
    Lines("Start GetLoader")
    
    # 1. XML 파일 존재 여부로 데이터 리스트 필터링
    xml_dir = meta.xml_dir
    train_list = train_list_all
    val_list = val_list_all
    test_list = test_list_all 

    if os.path.exists(xml_dir):
        xml_list_base = {os.path.splitext(file)[0] for file in os.listdir(xml_dir) if file.endswith(".xml")}
    else:
        Lines(f"경고: XML 디렉토리를 찾을 수 없습니다: {xml_dir}", bLines=False)
        xml_list_base = set()
        
    # 1.1. train_list 필터링
    original_train_len = len(train_list)
    train_list = [image for image in train_list if image in xml_list_base]
    if original_train_len != len(train_list):
        Lines(f"Train XML 파일이 없는 이미지 {original_train_len - len(train_list)}개 제거.")
    
    # 1.2. val_list 필터링
    original_val_len = len(val_list)
    val_list = [image for image in val_list if image in xml_list_base]
    if original_val_len != len(val_list):
        Lines(f"Validation XML 파일이 없는 이미지 {original_val_len - len(val_list)}개 제거.")

    # 2. Dataset 및 DataLoader 생성
    train_dataset = VOCDataset(image_dir=meta.image_dir, annotation_dir=meta.xml_dir, classes=meta.classes, image_list=train_list, transforms= transform)
    valid_dataset = VOCDataset(image_dir= meta.image_dir, annotation_dir=meta.xml_dir, classes= meta.classes, image_list=val_list, transforms=transform)
    test_dataset = TestDataset(image_dir=meta.image_dir, image_list=test_list, transforms=transform)
    
    Lines(f"Final dataset size: Train:{len(train_dataset)}, Valid:{len(valid_dataset)}, Test:{len(test_dataset)}")
    
    collate_fn = lambda x: tuple(zip(*x)) # 객체 탐지 모델용 collate_fn
    
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn, num_workers=meta.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn, num_workers=meta.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, collate_fn=collate_fn, num_workers=meta.num_workers)

    return train_loader, val_loader, test_loader


# ====================================================================
# 5. 실행 함수
# ====================================================================

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
    model_transform = model.get_default_transforms 
    
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
    
# 예시 실행 (SSD300VGG16Transfer)
Execute_Training(model_name="SSD", gubun="partial", epochs=1, lr=0.001)

# 예시 실행 (Faster R-CNN ResNet50)
Execute_Training(model_name="FasterRCNN", gubun="partial", epochs=1, lr=0.005)

# 예시 실행 (SSDLite MobileNetV3)
Execute_Training(model_name="SSDLite", gubun="freeze", epochs=1, lr=0.01)