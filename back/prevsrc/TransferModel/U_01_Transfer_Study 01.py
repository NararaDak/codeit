# ────────────────────────────────────────
# 4. Semantic Segmentation with DeepLabV3
# ────────────────────────────────────────
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import urllib.request
import os
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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
meta = weights.meta
# 5. 결과 시각화
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(meta["categories"])}
normalized_masks = torch.nn.functional.softmax(output, dim=1)
mask = normalized_masks[0].argmax(dim=0).byte().cpu().numpy()

# "person" 클래스에 해당하는 마스크만 추출
person_class_idx = sem_class_to_idx.get("person")
if person_class_idx is not None:
    person_mask = (mask == person_class_idx)  # (H_mask, W_mask), bool

    # 마스크를 원본 이미지 크기(img.size -> (width, height))로 리사이즈
    # NEAREST로 리샘플링하여 클래스 경계 유지
    mask_img = Image.fromarray(person_mask.astype('uint8') * 255)
    mask_img = mask_img.resize(img.size, resample=Image.NEAREST)
    person_mask_resized = np.array(mask_img).astype(bool)  # (H, W)

    # 원본 이미지 배열과 마스크를 채널 차원으로 브로드캐스트하여 적용
    masked_arr = np.array(img) * person_mask_resized[:, :, None]

    masked_img = Image.fromarray(masked_arr)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Original Image"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(masked_img); plt.title("Segmented Person"); plt.axis("off")
    MyShow("DeepLabV3 Segmentation Result",20)