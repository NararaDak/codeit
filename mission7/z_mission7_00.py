# === Code Cell 1 ===
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("zippyz/cats-and-dogs-breeds-classification-oxford-dataset")

# print("Path to dataset files:", path)
# *주석 처리된 코드 셀입니다.*


import torch
import datetime
import sys
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
#════════════════════════════════════════
# ▣ 유틸리티 함수 및 기본 설정
#════════════════════════════════════════

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
    # 1. 현재 시간 포맷팅
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    # 2. 호출한 함수 이름 가져오기
    # sys._getframe(1)은 OpLog을 호출한 함수의 프레임. 
    try:
        caller_name = sys._getframe(1).f_code.co_name
    except Exception:
        caller_name = "UnknownFunction"
        
    # 3. 로그 파일명 및 내용 포맷팅
    log_filename = f"{base_dir()}/op_log.txt"
    log_content = f"[{timestamp}] {caller_name}: {log}\n"
    # 4. 파일에 로그 추가 (append)
    try:
        # 'a' 모드는 파일이 없으면 생성하고, 있으면 기존 내용에 추가.
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(log_content)
    except Exception as e:
        print(f"로그 파일 쓰기 오류 발생: {e}")
Lines("Define Utility.")



# === Code Cell Separator ===
# === Code Cell 3 ===
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
#════════════════════════════════════════
# ▣ 유틸리티 함수 및 기본 설정
#════════════════════════════════════════

# GPU 설정
DEVICE_TYPE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Lines(DEVICE_TYPE)

base_dir = r"D:\05.gdrive\codeit\mission7\data\pet_data"
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


# 디렉토리 구하기.
#trainval_file_path, test_file_path,image_dir, xml_dir,df_trainval,df_test,trainval_list,test_list,xml_files =  GetDir()

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
import xml.etree.ElementTree as ET
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


transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
    ]
)
Lines("Define Transform.")

class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, classes, image_list, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.transforms = transforms
        self.image_files = image_list # 미리 필터링된 유효한 이미지 파일 리스트

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 및 XML 파일 경로 설정
        image_file = self.image_files[idx] + ".jpg"
        annotation_file = self.image_files[idx] + ".xml"
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, annotation_file)

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")

        # 어노테이션 로드
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

        # Tensor로 변환
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Transform 적용
        if self.transforms:
            image, boxes, labels = self.transforms(image, boxes, labels)

        target = {"boxes": boxes, "labels": labels}
        return image, target
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
Lines(f"Train 이미지 수: {len(train_list)}")
Lines(f"Validation 이미지 수: {len(valid_list)}")
Lines(f"Test 이미지 수: {len(test_list)}")

from torch.utils.data import DataLoader
# 클래스 정의
classes = ["background", "dog", "cat"]

# Train Dataset
train_dataset = VOCDataset(
    image_dir=image_dir,
    annotation_dir=xml_dir,
    classes=classes,
    image_list=train_list,  # Train 리스트 사용
    transforms=transform
)

# Validation Dataset
valid_dataset = VOCDataset(
    image_dir=image_dir,
    annotation_dir=xml_dir,
    classes=classes,
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
# ▣ 모델 준비
#════════════════════════════════════════
import torchvision
from torchvision.models.detection.ssd import SSD300_VGG16_Weights

# SSD 모델 불러오기
model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).to(DEVICE_TYPE)

# 클래스 개수에 맞게 출력 레이어 수정
num_classes = len(classes)  # background 포함
model.head.classification_head.num_classes = num_classes
Lines("Read SSD model")
import numpy as np
from sklearn.metrics import average_precision_score

def calculate_iou(box, boxes):
    """
    Calculate Intersection over Union (IoU) between a box and multiple boxes.

    Args:
        box (array): Single bounding box [x_min, y_min, x_max, y_max].
        boxes (array): Array of bounding boxes [[x_min, y_min, x_max, y_max], ...].

    Returns:
        array: IoU scores for each box in `boxes`.
    """
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    iou = intersection / union
    return iou


def calculate_ap(predictions, ground_truths, class_idx, iou_threshold=0.5):
    """
    특정 클래스에 대한 AP 계산.
    predictions: 모델의 예측 리스트 [{"boxes": [[x_min, y_min, x_max, y_max]], "labels": [label]}]
    ground_truths: 정답 리스트 [{"boxes": [[x_min, y_min, x_max, y_max]], "labels": [label]}]
    class_idx: 평가 대상 클래스 인덱스
    iou_threshold: IoU 기준값 (default=0.5)

    Returns:
        Average Precision (AP) 값
    """
    true_positives = []
    false_positives = []
    all_ground_truths = 0

    # 모든 예측과 정답을 순회
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = np.array(pred["boxes"])
        pred_labels = np.array(pred["labels"])
        gt_boxes = np.array(gt["boxes"])
        gt_labels = np.array(gt["labels"])

        # 현재 클래스에 해당하는 박스만 필터링
        pred_boxes = pred_boxes[pred_labels == class_idx]
        gt_boxes = gt_boxes[gt_labels == class_idx]

        all_ground_truths += len(gt_boxes)

        # IoU 계산
        detected = []
        for pred_box in pred_boxes:
            ious = []
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                ious.append(iou)

            if len(ious) > 0:
                max_iou_idx = np.argmax(ious)
                if ious[max_iou_idx] >= iou_threshold and max_iou_idx not in detected:
                    true_positives.append(1)
                    false_positives.append(0)
                    detected.append(max_iou_idx)
                else:
                    true_positives.append(0)
                    false_positives.append(1)
            else:
                false_positives.append(1)

    # Precision-Recall Curve 계산
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (all_ground_truths + 1e-6)

    # AP 계산
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def evaluate_model(predictions, ground_truths, classes):
    class_aps = []

    for class_idx, class_name in enumerate(classes[1:], start=1):
        true_positives = []
        scores = []
        num_ground_truths = 0

        for pred, gt in zip(predictions, ground_truths):
            # Filter for the current class
            pred_boxes = pred["boxes"][pred["labels"] == class_idx].cpu().numpy() if len(pred["boxes"]) > 0 else []
            pred_scores = pred["scores"][pred["labels"] == class_idx].cpu().numpy() if len(pred["scores"]) > 0 else []
            gt_boxes = gt["boxes"][gt["labels"] == class_idx].cpu().numpy() if len(gt["boxes"]) > 0 else []

            num_ground_truths += len(gt_boxes)

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue  # Skip if no predictions or ground truths for this class

            matched = np.zeros(len(gt_boxes), dtype=bool)
            for box, score in zip(pred_boxes, pred_scores):
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

        cum_true_positives = np.cumsum(true_positives)
        precision = cum_true_positives / (np.arange(len(true_positives)) + 1)
        recall = cum_true_positives / num_ground_truths

        ap = average_precision_score(true_positives, scores) if len(scores) > 0 else 0
        class_aps.append(ap)

    mAP = np.mean(class_aps)
    return mAP
Lines("Define eval functions.")
import torch
from torchvision.transforms import functional as F

from tqdm import tqdm # 진행 상황 시각화
import torch
import torch
from torchvision.transforms import functional as F

# Training + Validation Loop
# 
def DoTrain(numEpochs = 10, learnRate = 0.001):
    optimizer = torch.optim.SGD(model.parameters(), lr=learnRate, momentum=0.9, weight_decay=0.0005)
    lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(numEpochs):
        Lines(f"Epoch {epoch + 1}/{numEpochs} 시작")
        # Training Phase
        model.train()
        total_train_loss = 0
        for images, targets in tqdm(train_loader, desc="Training"):
            images = [img.to(DEVICE_TYPE) for img in images]
            targets = [{k: v.to(DEVICE_TYPE) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_train_loss += losses.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        Lines(f"Epoch {epoch + 1}/{numEpochs}, Train Loss: {avg_train_loss:.4f}")
        # Validation Phase
        model.eval()
        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(DEVICE_TYPE) for img in images]
                predictions = model(images)
                # 저장: 추론 결과와 Ground Truth
                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)

        # 성능 평가 (예: mAP 계산)
        mAP = evaluate_model(all_predictions, all_ground_truths, classes)
        Lines(f"Epoch {epoch + 1}/{numEpochs}, Validation mAP: {mAP:.4f}\n")
# 테스트로 시작 한다.
DoTrain(10,1) 