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
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from PIL import Image


# python /content/drive/MyDrive/codeit/mission7/src/z_mission7_06.py
#════════════════════════════════════════
# ▣ 환경 설정. 
#════════════════════════════════════════
BASE_DIR = r"D:\05.gdrive\codeit\mission7\data\pet_data"
#DBASE_DIR = "/content/drive/MyDrive/codeit/mission7/data/pet_data"
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
OpLog("Program started.",bLines=True)


#════════════════════════════════════════
# ▣ 메타 정보 클래스
#════════════════════════════════════════
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
        self._classes = ["background", "dog", "cat"]
        # 데이터프레임 로드
        try:
            self._df_trainval = pd.read_csv(self._trainval_file, sep="\s+", header=None)
            self._df_trainval.columns = ["Image", "ClassID", "Species", "BreedID"]
            
            self._df_test = pd.read_csv(self._test_file, sep="\s+", header=None)
            self._df_test.columns = ["Image", "ClassID", "Species", "BreedID"]

            self._trainval_list = self._df_trainval['Image'].tolist()
            xml_list = [os.path.splitext(file)[0] for file in os.listdir(self._xml_dir) if file.endswith(".xml")]
            # Train 이미지에 대해 XML 파일이 없는 경우 확인
            self._trainval_list = [image for image in self._trainval_list if image in xml_list]
            # Test에 사용될 이미지 파일 이름 리스트 생성
            self._test_list = self._df_test['Image'].tolist()

            # Test에 사용될 이미지 파일 이름 리스트 생성
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
    
    @property
    def trainval_list(self):
        return self._trainval_list
    @property
    def test_list(self):
        return self._test_list
    @property
    def classes(self):
        return self._classes
    
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
    print(f"XML 파일 개수: {len(meta.xml_files)}")
    # IoU 테스트
    boxA = [50, 50, 150, 150]
    boxB = [100, 100, 200, 200]
    iou = iou_box(boxA, boxB)
    Lines(f"IoU between boxA and boxB: {iou:.4f}")
    
    print(df_trainval.shape)
    df_trainval.head()
    print(df_test.shape)
    df_test.head()
    df_trainval['Species'].value_counts()
    df_test['Species'].value_counts()
    # Train과 Validation에 사용될 이미지 파일 이름 리스트 생성
    trainval_list = df_trainval['Image'].tolist()

    image_dir = meta.image_dir
    # Test에 사용될 이미지 파일 이름 리스트 생성
    test_list = df_test['Image'].tolist()
    # Train 데이터에서 예제 이미지 불러오기
    train_example_image_name = df_trainval["Image"].iloc[0]
    train_image_path = os.path.join(image_dir, f"{train_example_image_name}.jpg")

    # 이미지 읽기
    train_image = cv2.imread(train_image_path)
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)

    # Train 이미지 출력
    plt.imshow(train_image)
    plt.title(f"Train Image: {train_example_image_name}")
    plt.axis("off")
    plt.show()
    # Test 데이터에서 예제 이미지 불러오기
    test_example_image_name = df_test["Image"].iloc[0]
    test_image_path = os.path.join(image_dir, f"{test_example_image_name}.jpg")

    # 이미지 읽기
    test_image = cv2.imread(test_image_path)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    # Test 이미지 출력
    plt.imshow(test_image)
    plt.title(f"Test Image: {test_example_image_name}")
    plt.axis("off")
    plt.show()
    xml_dir = meta.xml_dir
    xml_list = [os.path.splitext(file)[0] for file in os.listdir(xml_dir) if file.endswith(".xml")]

    # Train 이미지에 대해 XML 파일이 없는 경우 확인
    missing_xml = [image for image in trainval_list if image not in xml_list]

    # Train 이미지에 대해 XML 파일이 있는 경우 확인
    trainval_list = [image for image in trainval_list if image in xml_list]

    # 결과 출력
    print(f"XML 파일이 없는 Train 이미지 수: {len(missing_xml)}")
    print(missing_xml)
   
#Test_Util_Functions()

#════════════════════════════════════════
# ▣ Xml 파싱 테스트 함수 
#═══════════════════════════════════════
def Test_xml_Parsing():
    meta = MyMeta()
    xml_dir = meta.xml_dir
    # 예제 XML 파일 경로
    xml_files =  [f for f in os.listdir(xml_dir) if f.endswith(".xml")]

    example_xml_file = os.path.join(xml_dir, xml_files[0])

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
    # XML 파일에서 Bounding Box와 클래스 정보 추출
    for obj in root.findall("object"):
        class_name = obj.find("name").text  # 클래스 이름
        bndbox = obj.find("bndbox")
        x_min = int(bndbox.find("xmin").text)
        y_min = int(bndbox.find("ymin").text)
        x_max = int(bndbox.find("xmax").text)
        y_max = int(bndbox.find("ymax").text)
        print(f"Class: {class_name}, Bounding Box: ({x_min}, {y_min}, {x_max}, {y_max})")
Test_xml_Parsing()

#════════════════════════════════════════
# ▣ 객체 검출 시각화 테스트 함수 
#════════════════════════════════════════
## XML 파일에서 어노테이션 정보를 추출하는 함수
def GetAnnotations():
    meta = MyMeta()
    xml_files = [file for file in os.listdir(meta.xml_dir) if file.endswith(".xml")]

    annotations = []
    for xml_file in xml_files:
        xml_path = os.path.join(meta.xml_dir, xml_file)
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

## 객체 검출 시각화 테스트 함수
def ViewObjectDectection():
    meta = MyMeta()
    annotations = GetAnnotations()
    # Train 데이터에서 예제 이미지 불러오기
    train_example_image_name = meta.df_trainval["Image"].iloc[0]
    train_image_path = os.path.join(meta.image_dir, f"{train_example_image_name}.jpg")

    # 이미지 읽기
    image = cv2.imread(train_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 해당 이미지의 어노테이션 가져오기
    annotations = [anno for anno in annotations if anno["image"] == f"{train_example_image_name}.jpg"]

    # Bounding Box 그리기
    for anno in annotations:
        x_min, y_min, x_max, y_max = anno["bbox"]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # 빨간색 박스
        cv2.putText(image, anno["class"], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 시각화
    plt.imshow(image)
    plt.axis("off")
    plt.show()
#ViewObjectDectection()

#════════════════════════════════════════
# ▣ 데이터 변환 함수 
#════════════════════════════════════════
def GetTransforms():
    transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
    ]
    )
    return transform
Lines("Define Transform Function.")


#════════════════════════════════════════
# ▣ 데이터셋 클래스
## Dataset 클래스를 상속하여 VOCDataset 클래스 정의
## 출력 : 이미지 텐서, 타겟 딕셔너리 (박스 좌표 텐서, 레이블 텐서)
'''
image, {
    "boxes": tensor([[xmin, ymin, xmax, ymax], ...]),
    "labels": tensor([0, 1, ...])
}
''' 
#═══════════════════════════════════════
class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, classes, image_list, transforms=None):
        ## 경로 및 클래스 정보 저장
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

        ## 전처리 변환.
        # Transform 적용
        if self.transforms:
            image, boxes, labels = self.transforms(image, boxes, labels)

        target = {"boxes": boxes, "labels": labels}
        return image, target
Lines("Define VOCDataset Class.")


#════════════════════════════════════════
# ▣ 테스트 및 추론 데이터셋 클래스
## Dataset 클래스를 상속.
## 출력 : 이미지 텐서, 타겟 딕셔너리 (박스 좌표 텐서, 레이블 텐서)
'''
image(실이미지), image_id(이미지 파일 이름)
''' 
#═══════════════════════════════════════
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
Lines("Define TestDataset Class.")

def GetTrainValidationSplit(df, test_size=0.3, random_state=42):
    meta = MyMeta()
    trainval_list = meta.trainval_list
    train_list, valid_list = train_test_split(trainval_list, test_size=0.3, random_state=42)
    # 결과 확인
    Lines(f"Train/Validation :{len(train_list)},{len(valid_list)}")
    return train_list, valid_list

def Test_TrainValidationSplit():
    meta = MyMeta()
    df_trainval = meta.df_trainval
    train_list, val_list = GetTrainValidationSplit(df_trainval, test_size=0.3, random_state=42)
    print(f"Train samples: {len(train_list)}")
    print(f"Validation samples: {len(val_list)}")
    print(f"test samples: {len(meta.test_list)}")
Test_TrainValidationSplit()

'''
[출력]
Train samples: 2576
Validation samples: 1104
test samples: 3669

'''