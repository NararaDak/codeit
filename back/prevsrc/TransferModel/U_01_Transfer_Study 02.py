#################################
## 배경합성
################################
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

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



# 1. 세그멘테이션 모델 불러오기
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()
preprocess = weights.transforms()

# 2. 이미지 다운로드 및 불러오기 (안전한 다운로드 + 읽기 폴백)
base_path = "D:/01.project/코드잇/src/TransferModel/synthesis_images"
os.makedirs(base_path, exist_ok=True)

# Unsplash 등에서 직접 접근 실패하는 경우가 있어 파라미터를 추가
img_url = "https://images.unsplash.com/photo-1583512603805-3cc6b41f3edb"
background_url = "https://images.unsplash.com/photo-1507525428034-b723a996f329"

img_path = os.path.join(base_path, "test.jpg")
background_path = os.path.join(base_path, "background.jpg")

def download_image(url, path):
    # Unsplash 이미지는 query 파라미터가 있으면 안정적이라 파라미터를 붙임
    url_with_params = url + "?auto=format&fit=crop&w=1350&q=80"
    try:
        req = urllib.request.Request(url_with_params, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Downloaded: {path}")
        return True
    except Exception as e:
        print(f"Failed to download {url} -> {e}")
        return False

# 시도해서 없으면 경고, 로컬파일이 있으면 사용
download_image(img_url, img_path)
download_image(background_url, background_path)

# 다운로드 실패로 파일이 없을 수 있으므로 존재 여부 검사하고 대체 처리
if not os.path.exists(img_path):
    raise FileNotFoundError(f"원본 이미지가 없습니다: {img_path}")

if not os.path.exists(background_path):
    # 배경 이미지가 없으면 원본 이미지를 배경으로 사용하거나 단색 배경 생성
    print(f"배경 이미지가 없어서 원본 이미지를 배경으로 사용합니다: {background_path}")
    background_path = img_path
    # 또는 단색 배경을 사용하려면 아래처럼 설정할 수 있음:
    # background_path = None

def load_image_rgb(path):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    img_cv = cv2.imread(path)
    if img_cv is not None:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    try:
        pil_img = Image.open(path).convert("RGB")
        arr = np.array(pil_img)
        print(f"Loaded with PIL fallback: {path}")
        return arr
    except Exception as e:
        print(f"이미지 읽기 실패(cv2, PIL 모두 실패): {path} -> {e}")
        return None

# 안전하게 불러오기
img = load_image_rgb(img_path)
if img is None:
    raise RuntimeError("원본 이미지를 불러오지 못했습니다.")

background = load_image_rgb(background_path)
if background is None:
    # 배경을 못 불러오면 원본 크기의 단색 배경 생성
    print("배경을 불러오지 못해 단색 배경을 생성합니다.")
    background = np.ones_like(img) * 255  # 흰색 배경

# 배경을 원본 크기로 맞춤
background = cv2.resize(background, (int(img.shape[1]), int(img.shape[0])))

# 3. 추론
pil_img = Image.fromarray(img)
input_tensor = preprocess(pil_img).unsqueeze(0)
with torch.no_grad():
    out = model(input_tensor)["out"][0]
pred = out.argmax(0).byte().cpu().numpy()
pred_resized = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# 4. 사람(15) + 강아지(12) 마스크
mask = ((pred_resized == 15) | (pred_resized == 12)).astype(np.uint8)

# 5. 합성
mask_3ch = np.repeat(mask[:, :, None], 3, axis=2)
combined = img * mask_3ch + background * (1 - mask_3ch)

# 6. 시각화 (원본 / 세그멘테이션 / 배경 / 합성)
plt.figure(figsize=(20,6))

plt.subplot(1,4,1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(pred_resized, cmap="tab20")  # 세그멘테이션 마스크
plt.title("Segmentation Mask")
plt.axis("off")

plt.subplot(1,4,3)
plt.imshow(background)
plt.title("Background")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(combined.astype(np.uint8))
plt.title("Combined Result")
plt.axis("off")

plt.show()