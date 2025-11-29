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
# Use TkAgg for interactive display (instead of non-interactive Agg)
# This allows plt.show() to actually display windows and block until user closes them
matplotlib.use('TkAgg')
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

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDClassificationHead
from ultralytics import YOLO
import yaml

# 사전작업 함수 
def PreJob():
    # 이미지 베이스 경로
    BASE_DIR = r"D:\05.gdrive\codeit\mission7\data\pet_data" 
    IMAGE_DIR = f"{BASE_DIR}/images/images"

    # 학습 이미지 경로 (BASE_IMAGE_DIR 하위의 train 폴더 가정)
    TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train")
    # 검증 이미지 경로 (BASE_IMAGE_DIR 하위의 val 폴더 가정)
    VAL_IMAGE_DIR = os.path.join(IMAGE_DIR, "val")

    # 클래스 이름
    YOLO_CLASS_NAMES = ['dog', 'cat']
    NUM_CLASSES = len(YOLO_CLASS_NAMES)

    # YAML 파일 저장 경로
    YAML_FILE = f"{BASE_DIR}/annotations/pets.yaml"

    yaml_data = {
        'train': TRAIN_IMAGE_DIR,
        'val': VAL_IMAGE_DIR,
        'nc': NUM_CLASSES,
        'names': YOLO_CLASS_NAMES
    }

    # YAML 파일로 저장
    with open(YAML_FILE, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)
    return YAML_FILE

def Train():
    # YOLOv8n 사전 훈련된 모델(.pt)을 로드합니다.
    yaml_file = PreJob()
    model = YOLO('yolov8n.pt') 
    # 학습 실행
    # data: 위에서 작성한 pets.yaml 파일 경로
    # epochs: 학습 횟수
    # imgsz: 이미지 크기 (640 또는 더 크게 설정 가능)
    results = model.train(
        data= yaml_file,
        epochs=50, 
        imgsz=640, 
        name='yolov8n_pets_train'
    )

