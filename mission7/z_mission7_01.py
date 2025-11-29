# === Code Cell 1 ===
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("zippyz/cats-and-dogs-breeds-classification-oxford-dataset")

# print("Path to dataset files:", path)
# *주석 처리된 코드 셀입니다.*

import xml.etree.ElementTree as ET
import torch
import datetime
import sys
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision import models
import os
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
#════════════════════════════════════════
# ▣ 유틸리티 함수 및 기본 설정
#════════════════════════════════════════
base_dir = r"D:\05.gdrive\codeit\mission7\data\pet_data"
#base_dir = "/content/drive/MyDrive/codeit/mission7/data/pet_data"

## 라인 구분 함수
def Lines(text="", count=100):
    print("═" * count)
    if text != "":
        print(f"{text}")
        print("═" * count)

## 현재 시간 가져오기 함수
def GetCurTime():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time
    
def PrintCurTime():
    Lines(f"[{GetCurTime()}]")

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
    log_filename = f"{base_dir}/op_log.txt"
    log_content = f"[{GetCurTime()}] {caller_name}: {log}\n"
    # 4. 파일에 로그 추가 (append)
    try:
        # 'a' 모드는 파일이 없으면 생성하고, 있으면 기존 내용에 추가.
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(log_content)
    except Exception as e:
        print(f"로그 파일 쓰기 오류 발생: {e}")
Lines("Define Utility.")

OpLog("Start Program")
#════════════════════════════════════════
# ▣ 전역 변수 설정 
#════════════════════════════════════════
# GPU 설정
DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Lines(DEVICE_TYPE)

# 파일 경로 설정
trainval_file_path = os.path.join(base_dir, "annotations", "annotations", "trainval.txt")
test_file_path = os.path.join(base_dir, "annotations", "annotations", "test.txt")

# 이미지, Annotation 경로 설정
image_dir = os.path.join(base_dir, "images", "images")
xml_dir = os.path.join(base_dir, "annotations", "annotations", "xmls")

# Train/Validation 파일 읽기
df_trainval = pd.read_csv(trainval_file_path, sep="\s+", header=None)
df_trainval.columns = ["Image", "ClassID", "Species", "BreedID"]

# Test 파일 읽기
df_test = pd.read_csv(test_file_path, sep="\s+", header=None)
df_test.columns = ["Image", "ClassID", "Species", "BreedID"]

# 데이터 크기 확인
Lines(f"Train/Validation 데이터 수: {len(df_trainval)}")
Lines(f"Test 데이터 수: {len(df_test)}")

# Annotation 개수 확인
xml_files = [file for file in os.listdir(xml_dir) if file.endswith(".xml")]
Lines(f"XML 파일 개수: {len(xml_files)}")

Lines(df_trainval.shape)
df_trainval.head()

Lines(df_test.shape)
df_test.head()

df_trainval['Species'].value_counts()

df_test['Species'].value_counts()

# Train과 Validation에 사용될 이미지 파일 이름 리스트 생성
trainval_list = df_trainval['Image'].tolist()

# Test에 사용될 이미지 파일 이름 리스트 생성
test_list = df_test['Image'].tolist()
Lines(f"trainval_list:{trainval_list}")
Lines(f"test_list:{test_list}")
#return trainval_file_path, test_file_path, image_dir, xml_dir,df_trainval,df_test, trainval_list,test_list,xml_files

#════════════════════════════════════════
# ▣ View Test 
#════════════════════════════════════════
def ViewTest():
    train_example_image_name = df_trainval["Image"].iloc[0]
    train_image_path = os.path.join(image_dir, f"{train_example_image_name}.jpg")

    # 이미지 읽기
    train_image = cv2.imread(train_image_path)
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)

    # Train 이미지 출력
    plt.imshow(train_image)
    plt.title(f"Train Image: {train_example_image_name}")
    plt.axis("off")
    plt.show(block = False)
    plt.pause(3)
    plt.close()

     # Test 데이터에서 예제 이미지 불러오
    test_example_image_name = df_test["Image"].iloc[0]
    test_image_path = os.path.join(image_dir, f"{test_example_image_name}.jpg")

    # 이미지 읽기
    test_image = cv2.imread(test_image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    # Test 이미지 출력
    plt.imshow(test_image)
    plt.title(f"Test Image: {test_example_image_name}")
    plt.axis("off")
    plt.show(block = False)
    plt.pause(3)
    plt.close()
ViewTest()

#════════════════════════════════════════
# ▣ XML 탐구. 
#════════════════════════════════════════
def ExploreXml(trainvalList,xmlDir):
    # XML 파일 이름 가져오기 (확장자 제거)
    xml_list = [os.path.splitext(file)[0] for file in os.listdir(xmlDir) if file.endswith(".xml")]

    # Train 이미지에 대해 XML 파일이 없는 경우 확인
    missing_xml = [image for image in trainvalList if image not in xml_list]

    # Train 이미지에 대해 XML 파일이 있는 경우 확인
    trainvalList = [image for image in trainvalList if image in xml_list]

    # 결과 출력
    print(f"XML 파일이 없는 Train 이미지 수: {len(missing_xml)}")
    print(missing_xml)

    # 예제 XML 파일 경로
    example_xml_file = os.path.join(xmlDir, xml_files[0])
    # XML 파일 읽기 및 파싱
    tree = ET.parse(example_xml_file)
    root = tree.getroot()

    # 재귀적으로 모든 태그와 데이터 출력 함수
    def print_all_elements(element, indent=""):
        print(f"{indent}{element.tag}: {element.text}")
        for child in element:
            print_all_elements(child, indent + "  ")

    # XML 구조 탐색
    print_all_elements(root)
    for obj in root.findall("object"):
        class_name = obj.find("name").text  # 클래스 이름
        bndbox = obj.find("bndbox")
        x_min = int(bndbox.find("xmin").text)
        y_min = int(bndbox.find("ymin").text)
        x_max = int(bndbox.find("xmax").text)
        y_max = int(bndbox.find("ymax").text)

        print(f"Class: {class_name}, Bounding Box: ({x_min}, {y_min}, {x_max}, {y_max})")

ExploreXml(trainval_list,xml_dir)

def AllXml():
    for xml_file in xml_files:
        annotations = []
        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        image_name = root.find("filename").text  # 이미지 파일 이름

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            bndbox = obj.find("bndbox")
            x_min = int(bndbox.find("xmin").text)
            y_min = int(bndbox.find("ymin").text)
            x_max = int(bndbox.find("xmax").text)
            y_max = int(bndbox.find("ymax").text)

            annotations.append({
                "image": image_name,
                "class": class_name,
                "bbox": [x_min, y_min, x_max, y_max]
            })
    return annotations 
annotations_all = AllXml()

#════════════════════════════════════════
# ▣ 객체 보기
#════════════════════════════════════════
def ViewObjectDection(dfTrainval,annotationsAll):
    train_example_image_name = dfTrainval["Image"].iloc[0]
    train_image_path = os.path.join(image_dir, f"{train_example_image_name}.jpg")

    # 이미지 읽기
    image = cv2.imread(train_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 해당 이미지의 어노테이션 가져오기
    annotationsAll = [anno for anno in annotationsAll if anno["image"] == f"{train_example_image_name}.jpg"]

    # Bounding Box 그리기
    for anno in annotationsAll:
        x_min, y_min, x_max, y_max = anno["bbox"]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # 빨간색 박스
        cv2.putText(image, anno["class"], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 시각화
    plt.imshow(image)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

ViewObjectDection(df_trainval,annotations_all)
#════════════════════════════════════════
# ▣ Loader 생성.
#════════════════════════════════════════
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
    ]
)
Lines("Define Transform.")

def Valid_annotations(image_list, annotation_dir):
    """
    이미지 리스트를 입력받아, 대응하는 XML 어노테이션 파일이 존재하는
    유효한 이미지 파일 리스트만 반환합니다.
    """
    filtered_list = []
    
    # tqdm 라이브러리를 사용하면 필터링 진행 상황을 볼 수 있어 편리합니다. (선택 사항)
    # from tqdm.auto import tqdm
    # for image_file in tqdm(image_list, desc="Filtering Dataset"): 
    
    for image_file in image_list:
        # 1. 파일 이름에서 확장자를 분리하고 XML 확장자로 대체합니다.
        # 예: "00001.jpg" -> "00001.xml"
        annotation_file_name = os.path.splitext(image_file)[0] + '.xml' 
        annotation_path = os.path.join(annotation_dir, annotation_file_name)
        
        # 2. XML 파일이 실제로 존재하는지 확인합니다.
        if os.path.exists(annotation_path):
            filtered_list.append(image_file)
        else:
            OpLog(f"제외: {image_file}에 대응하는 어노테이션({annotation_dir}/{annotation_file_name})이 없습니다.",False)
    print(f"원본 이미지 개수: {len(image_list)} -> 유효 이미지 개수: {len(filtered_list)}")
    return filtered_list

class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, classes, image_list, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.transforms = transforms
        image_list = Valid_annotations(image_list,annotation_dir)
        self.image_files = image_list # 미리 필터링된 유효한 이미지 파일 리스트

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 파일 이름 정의 및 경로 설정
        image_file_name = self.image_files[idx] + ".jpg"
        base_name = os.path.splitext(image_file_name)[0]
        annotation_file_name = base_name + ".xml"
        image_path = os.path.join(self.image_dir, image_file_name)
        annotation_path = os.path.join(self.annotation_dir, annotation_file_name)

        # 1. 오류 발생 시 반환할 기본(Empty) 텐서 정의
        empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
        empty_labels = torch.zeros((0,), dtype=torch.int64)
        target_default = {"boxes": empty_boxes, "labels": empty_labels}
        # SSD 모델을 고려하여 300x300 크기의 더미 이미지 텐서 생성
        dummy_image = torch.zeros((3, 300, 300), dtype=torch.float32) 

        try:
            # 2. 이미지 로드 (기존 로직)
            image = Image.open(image_path).convert("RGB")

            # 어노테이션 로드 (기존 로직)
            boxes = []
            labels = []
            tree = ET.parse(annotation_path)
            root = tree.getroot()

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

            # Tensor로 변환 (boxes가 비어있을 경우를 명시적으로 처리)
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else empty_boxes
            labels = torch.tensor(labels, dtype=torch.int64) if labels else empty_labels
            
            # Transform 적용 (원본 코드 유지)
            if self.transforms:
                image, boxes, labels = self.transforms(image, boxes, labels)

            target = {"boxes": boxes, "labels": labels}
            return image, target

        except Exception as e:
            # 3. 오류 발생 시 경고 로그를 출력하고 유효한 빈 데이터를 반환
            print(f"Warning: Failed to load/parse sample index {idx} ({base_name}). Error: {e}. Returning dummy data.")
            # 오류 발생 시 유효한 더미 이미지 텐서와 빈 타겟 딕셔너리 반환
            return dummy_image, target_default
Lines("Define VOCDataset")

class TestDataset(Dataset):
    def __init__(self, image_dir, image_list, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_files = image_list  # 테스트 이미지 리스트 (확장자 없음)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 파일 경로
        image_file = self.image_files[idx] + ".jpg"
        image_path = os.path.join(self.image_dir, image_file)

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # Transform 적용
        if self.transforms:
            image = self.transforms(image)

        return image, self.image_files[idx]  # 이미지와 파일 이름 반환
Lines("Define class TestDataset.")

train_list, valid_list = train_test_split(trainval_list, test_size=0.3, random_state=42)

# 결과 확인
OpLog(f"Train 이미지 수: {len(train_list)}")
OpLog(f"Validation 이미지 수: {len(valid_list)}")
OpLog(f"Test 이미지 수: {len(test_list)}")

from torch.utils.data import DataLoader
# 클래스 정의
classes_type = ["background", "dog", "cat"]

# Train Dataset
train_dataset = VOCDataset(
    image_dir=image_dir,
    annotation_dir=xml_dir,
    classes=classes_type,
    image_list=train_list,  # Train 리스트 사용
    transforms=transform
)

# Validation Dataset
valid_dataset = VOCDataset(
    image_dir=image_dir,
    annotation_dir=xml_dir,
    classes=classes_type,
    image_list=valid_list,  # Validation 리스트 사용
    transforms=transform
)

# Test Dataset 생성
test_dataset = TestDataset(
    image_dir=image_dir,  # 테스트 이미지 디렉토리
    image_list=test_list,       # 테스트 이미지 리스트 (확장자 없는 이름)
    transforms=transform  # 필요하면 Transform 적용
)


# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 데이터 크기 출력
Lines(f"Train 데이터셋 크기: {len(train_dataset)}")
Lines(f"Validation 데이터셋 크기: {len(valid_dataset)}")
Lines(f"Test 데이터셋 크기: {len(test_dataset)}")

#════════════════════════════════════════
# ▣ 평가(Metrics) 함수 모듈
#════════════════════════════════════════
def calculate_iou(box, boxes):
    """IoU 계산 (SSD 평가용)"""
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / union

def calculate_map(model, dataloader, device, classes):
    """Detection 모델 mAP 평가"""
    model.eval()
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            # targets는 dict 리스트, 텐서만 GPU 이동
            targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            preds = model(images)
            
            # CPU로 이동하여 저장
            for p in preds:
                all_predictions.append({k: v.cpu().numpy() for k, v in p.items()})
            for t in targets: # 원본 target 사용 (CPU 가정)
                all_ground_truths.append({k: v.cpu().numpy() for k, v in t.items()})

    class_aps = []
    # 0번(Background) 제외
    for class_idx in range(1, len(classes)):
        true_positives = []
        scores = []
        num_gt = 0
        
        for pred, gt in zip(all_predictions, all_ground_truths):
            pred_mask = pred['labels'] == class_idx
            gt_mask = gt['labels'] == class_idx
            
            p_boxes = pred['boxes'][pred_mask]
            p_scores = pred['scores'][pred_mask]
            g_boxes = gt['boxes'][gt_mask]
            
            num_gt += len(g_boxes)
            if len(p_boxes) == 0: continue
            if len(g_boxes) == 0:
                true_positives.extend([0] * len(p_boxes))
                scores.extend(p_scores)
                continue
                
            matched = np.zeros(len(g_boxes), dtype=bool)
            for box, score in zip(p_boxes, p_scores):
                ious = calculate_iou(box, g_boxes)
                max_iou_idx = np.argmax(ious) if len(ious) > 0 else -1
                if max_iou_idx >= 0 and ious[max_iou_idx] >= 0.5 and not matched[max_iou_idx]:
                    true_positives.append(1)
                    matched[max_iou_idx] = True
                else:
                    true_positives.append(0)
                scores.append(score)
                
        if len(scores) == 0:
            class_aps.append(0)
        else:
            indices = np.argsort(-np.array(scores))
            tp = np.array(true_positives)[indices]
            sc = np.array(scores)[indices]
            ap = average_precision_score(tp, sc) if num_gt > 0 else 0
            class_aps.append(ap)
            
    mAP = np.mean(class_aps) if class_aps else 0
    return {'mAP': mAP}

def calculate_accuracy(model, dataloader, device):
    """Classification 모델 정확도 평가"""
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return {'accuracy': correct / total, 'avg_loss': running_loss / total}

#════════════════════════════════════════
# ▣ 모델 클래스 정의 (학습 설정 포함)
#════════════════════════════════════════
## 기본 Class.
class BasicModel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.real_learn_rate = 0.0
    def GetMyName(self): return "BasicModel"
    def setup_training(self): raise NotImplementedError

## SSD.
class SSDTransfer(BasicModel):
    def __init__(self, num_classes, gubun="freeze", learn_rate=1e-3, backbone_lr_ratio=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gubun = gubun
        self.learn_rate = learn_rate
        self.backbone_lr_ratio = backbone_lr_ratio
        
        Lines(f"Initializing SSDTransfer - Ratio: {self.backbone_lr_ratio}")
        weights = SSD300_VGG16_Weights.DEFAULT
        self.model = ssd300_vgg16(weights=weights)
        self.model.head.classification_head.num_classes = num_classes 

    def GetMyName(self): return "SSD300"
    def GetGubun(self):  return self.gubun
    def forward(self, images, targets=None):
        if self.training: return self.model(images, targets)
        return self.model(images)
            
    def setup_training(self):
        backbone_params = []
        head_params = []
        
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
        
        params_group = [{'params': head_params, 'lr': self.learn_rate}]
        
        if self.gubun != "freeze" and backbone_params:
            eff_lr = self.learn_rate * self.backbone_lr_ratio
            params_group.append({'params': backbone_params, 'lr': eff_lr})
            Lines(f" >> Optimizer Setup: Head LR={self.learn_rate}, Backbone LR={eff_lr}")
        optimizer = torch.optim.SGD(params_group, lr=self.learn_rate, momentum=0.9, weight_decay=0.0005)
        return optimizer


#════════════════════════════════════════
# ▣ Trainer 클래스.
#════════════════════════════════════════
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, model_type, checkpoint_dir, classes, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader # Test Loader 저장 (마지막에 사용)
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type
        self.checkpoint_dir = checkpoint_dir
        self.classes = classes
        self.config = config 
        self.result_csv = os.path.join(checkpoint_dir, "training_results.csv")
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_to_csv(self, metrics, epoch, current_lr, dataset_name):
        """CSV 저장 함수"""
        data = {
            'Timestamp': [GetCurTime()],
            'Model': [self.model_type],
            'Strategy': [self.config.get('gubun')],
            'DataSet': [dataset_name], 
            'Epoch': [epoch],
            'Base_LR': [self.config.get('base_lr')],       
            'Ratio': [self.config.get('ratio', 'N/A')],    
            'Current_Head_LR': [current_lr],                
            'Current_Backbone_LR': [current_lr * self.config.get('ratio', 0) if self.config.get('ratio') else 'N/A'] 
        }
        
        for k, v in metrics.items():
            data[k] = [v]
            
        df = pd.DataFrame(data)
        
        if os.path.exists(self.result_csv):
            try:
                existing_df = pd.read_csv(self.result_csv)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(self.result_csv, index=False)
            except pd.errors.EmptyDataError:
                df.to_csv(self.result_csv, index=False)
        else:
            df.to_csv(self.result_csv, index=False)
            
        print(f" >> CSV Saved ({dataset_name}): Epoch {epoch}, LR {current_lr:.6f}")

    def train_one_epoch(self,epoch):
        self.model.train()
        running_loss = 0.0
        loop = tqdm(self.train_loader, desc="Train",disable=True)
        index = 0
        for batch_idx, (data, targets) in enumerate(loop):
            self.optimizer.zero_grad()
            
            images = [d.to(self.device) for d in data]
            t_targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, t_targets)
            loss = sum(loss for loss in loss_dict.values())
           
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            logs = f"{self.model.GetMyName()}_{self.model.GetGubun()}, Epoch: {epoch}, Batch: {batch_idx + 1},[{index}/{len(train_loader)}], Loss: {loss.item():.4f}"
            OpLog(logs,False)
            print(logs,end="\r")
            index += 1
            
        return running_loss / len(self.train_loader)

    def validate(self, dataloader, desc="Validation"):
        # 평가 수행
        return calculate_map(self.model, dataloader, self.device, self.classes)

    def fit(self, num_epochs): 
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        OpLog(f"Start Training - Scheduler: StepLR (num_epochs={num_epochs}, step=3, gamma=0.1)")
        
        current_lr = self.config.get('base_lr')

        for epoch in range(1, num_epochs + 1):
            current_lr = scheduler.get_last_lr()[0]
            
            # 1. 학습 (Train)
            train_loss = self.train_one_epoch(epoch)
            
            # 2. 검증 (Validation) - 모델 선정의 기준
            OpLog(f"Evaluating on Validation Set (Epoch {epoch})")
            val_metrics = self.validate(self.val_loader, desc="Validation")
            val_metrics['train_loss'] = train_loss
            self.save_to_csv(val_metrics, epoch, current_lr, "Validation")
            
            # 스케줄러 및 체크포인트
            scheduler.step()
            
            save_path = f"{self.checkpoint_dir}/{self.model_type}_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'metrics': val_metrics,
                'config': self.config
            }, save_path)
            OpLog(f"Checkpoint Saved: {save_path}")
            
        # ====================================================
        # 3. 최종 테스트 (Test) - 모든 학습 종료 후 단 1회 실행
        # ====================================================
        OpLog("Training Finished. Starting Final Evaluation on Test Set...")
        final_test_metrics = self.validate(self.test_loader, desc="Test_Final")
        
        # 최종 결과 CSV 저장 (Epoch은 완료된 상태이므로 num_epochs로 기록)
        self.save_to_csv(final_test_metrics, num_epochs, current_lr, "Test_Final")
        OpLog(f"Final Test Results Saved to {self.result_csv}")

#════════════════════════════════════════
# ▣ 실행 함수.
#════════════════════════════════════════
def MakeModel(model_type, num_classes, gubun, lr, ratio=0.1):
    if model_type == "SSD":
        model = SSDTransfer(num_classes, gubun, lr, backbone_lr_ratio=ratio)
    else:
        raise ValueError(f"Unknown Model Class: {model_type}")
        
    optimizer = model.setup_training()
    return model, optimizer

def Execute_fn(whatClass, learn_rate, num_epochs, gubun, backbone_ratio=0.1):
    # 1. 모델 이름 및 경로 설정을 위한 임시 호출
    temp_model, _ = MakeModel(whatClass, len(classes_type), gubun, learn_rate, ratio=backbone_ratio)
    model_name = temp_model.GetMyName()
    del temp_model 

    checkpoint_dir = os.path.join(base_dir, "modelfiles", f"checkpoints_{model_name}_{gubun}_{num_epochs}_{learn_rate}")
    
    OpLog(f"Start Execution: {model_name} / {gubun} / LR={learn_rate} / Ratio={backbone_ratio}")

    # 2. 실제 모델 생성
    model, optimizer = MakeModel(whatClass, len(classes_type), gubun, learn_rate, ratio=backbone_ratio)
    
    # 3. Trainer 설정
    config_data = {
        'gubun': gubun,
        'base_lr': learn_rate,
        'ratio': backbone_ratio
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device= DEVICE_TYPE,
        model_type=whatClass,
        checkpoint_dir=checkpoint_dir,
        classes=classes_type,
        config=config_data
    )
    
    # 4. 학습 시작 (중간 Test 없음, 오직 마지막에 1회 수행)
    trainer.fit(num_epochs)
    
    OpLog(f"End Execution for {model_name}_{gubun}")

#════════════════════════════════════════
# ▣ 함수 실행.
#════════════════════════════════════════
Execute_fn("SSD", 1, 2, "partial", backbone_ratio=0.1)
