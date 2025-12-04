import torchvision
print(torchvision.__version__)
from torchvision import models
print(dir(models)[:5])

# help(models.AlexNet)

import inspect
print(inspect.signature(models.AlexNet))

#────────────────────────────────────────
# 유틸리티 함수
#────────────────────────────────────────
def Lines(linecount=90):
    print("─" * linecount + "\n")
    
def MyShow(title = "그림",seconds=3):
    plt.title(title)
    plt.show(block=False)
    plt.pause(seconds)
    plt.close()


## ImageNet 클래스 라벨 가져와 .txt로 저장하기
import urllib.request
imagenet_classes_text = "D:/01.project/코드잇/src/TransferModel/data/imagenet_classes.txt"
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, imagenet_classes_text)

## 저장된 txt파일 읽어서 라인단위로 분리해 classes변수에 넣기

with open(imagenet_classes_text) as f:
      classes = [line.strip() for line in f.readlines()]

Lines()
print(classes[:5])
Lines()

import torch
from torchvision import models, transforms
from PIL import Image

import matplotlib.pyplot as plt

# 1. 사전학습된 AlexNet 불러오기
model = models.alexnet(weights="IMAGENET1K_V1")
model.eval()

# 2. 전처리 파이프라인 (ImageNet 규격)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet 평균
        std=[0.229, 0.224, 0.225]     # ImageNet 표준편차
    ),
])


# 3. 예시 이미지 불러오기 (로컬 파일도 가능)
base_path = "D:/01.project/코드잇/src/TransferModel/predict_image"
img_list=[f"{base_path}/추론cat.jpg", f"{base_path}/추론dog.jpg", f"{base_path}/추론ship.jpg", f"{base_path}/추론cat2.jpg"]
for imgfile in img_list:
  img = Image.open(imgfile)
  x = preprocess(img).unsqueeze(0)
  print(x.shape) # 모델에서 요구하는 1, 3,224,224

  # 4. 추론
  with torch.no_grad():
      outputs = model(x)
      _, predicted = outputs.max(1)


  print("\n ->예측된 클래스번호:", predicted.item())
  print(" ->예측된 클래스:", classes[predicted.item()])
  plt.figure(figsize=(3,3))
  plt.axis('off')   # x, y 축 모두 숨김
  plt.imshow(img)
  MyShow("Predicted: " + classes[predicted.item()])


  
from torchvision.models.detection import  FasterRCNN_ResNet50_FPN_Weights

# 1. 가중치 불러오기
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

# 2. 메타데이터 확인
meta = weights.meta
print(meta.keys())
print(len(meta["categories"]))   # 학습된 클래스 개수 (COCO: 91개)
print(meta["categories"]) 

import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# 1. 모델 & 가중치 불러오기
base_path = "D:/01.project/코드잇/src/TransferModel/preResNet5dict_image"
os.makedirs(base_path, exist_ok=True) # Create the directory if it doesn't exist
# # 객체 탐지(Object Detection)용 테스트 이미지 다운로
os.makedirs(base_path, exist_ok=True) # Create the directory if it doesn't exist

detection_img_url = "https://ultralytics.com/images/zidane.jpg"
detection_img_path = os.path.join(base_path, "detection_test.jpg") # base_path는 위에서 정의됨
urllib.request.urlretrieve(detection_img_url, detection_img_path)


weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# 2. COCO 클래스 이름
categories = weights.meta["categories"]

# 3. 이미지 불러오기
img = Image.open(detection_img_path).convert("RGB")

# 4. 전처리
transform = weights.transforms()
img_tensor = transform(img)

# 5. 추론
with torch.no_grad():
    preds = model([img_tensor])[0]

# 6. 바운딩박스 + 라벨 출력
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
except:
    font = ImageFont.load_default()

for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
    if score < 0.5:
        continue
    x1, y1, x2, y2 = box

    # 박스
    draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)

    # 레이블 + 점수
    text = f"{categories[label]}: {score:.2f}"
    bbox = font.getbbox(text)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 텍스트 배경 박스
    draw.rectangle(((x1, y1 - text_h), (x1 + text_w, y1)), fill="red")

    # 텍스트 출력
    draw.text((x1, y1 - text_h), text, fill="white", font=font)

# 7. 출력
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis("off")
MyShow("Faster R-CNN Detection Result")

from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

# 가중치 불러오기
weights = DeepLabV3_ResNet50_Weights.DEFAULT
meta = weights.meta
print(meta.keys())
print(len(meta["categories"]))  # 학습된 클래스 레이블 목록
print(meta["categories"])

Lines()

# ────────────────────────────────────────
# 4. Semantic Segmentation with DeepLabV3
# ────────────────────────────────────────
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. 모델 & 가중치 불러오기
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights)
model.eval()
base_path = "D:/01.project/코드잇/src/TransferModel/ResNet50_Weights"
os.makedirs(base_path, exist_ok=True)
# 2. 전처리 파이프라인
preprocess = weights.transforms()

# 3. 이미지 다운로드 및 불러오기
segmentation_img_url = "https://www.learnopencv.com/wp-content/uploads/2021/01/person-segmentation.jpeg"
segmentation_img_path = os.path.join(base_path, "segmentation_test.jpg") # Using base_path from object detection part

# Add a User-Agent header to avoid HTTP 403 Forbidden error
req = urllib.request.Request(segmentation_img_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(segmentation_img_path, 'wb') as out_file:
    data = response.read()
    out_file.write(data)

img = Image.open(segmentation_img_path)
x = preprocess(img).unsqueeze(0)

# 4. 추론
with torch.no_grad():
    output = model(x)['out']

# 5. 결과 시각화
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(meta["categories"])}
normalized_masks = torch.nn.functional.softmax(output, dim=1)
mask = normalized_masks[0].argmax(dim=0).byte().cpu().numpy()

# "person" 클래스에 해당하는 마스크만 추출
person_class_idx = sem_class_to_idx.get("person")
if person_class_idx is not None:
    person_mask = (mask == person_class_idx)
    masked_img = Image.fromarray(np.array(img) * person_mask[:, :, None])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Original Image"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(masked_img); plt.title("Segmented Person"); plt.axis("off")
    MyShow("DeepLabV3 Segmentation Result")