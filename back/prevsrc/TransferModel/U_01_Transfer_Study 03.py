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

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


def load_image_rgb_safe(path):
    """
    경로가 유효한지 검사하고 cv2로 먼저 읽고 실패하면 PIL로 폴백하여 RGB numpy 배열을 반환합니다.
    파일이 없거나 읽기 실패 시 None을 반환합니다.
    """
    # 절대 경로 변환
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    if not os.path.exists(path):
        print(f"이미지 파일이 없습니다: {path}")
        return None

    # cv2로 시도 (BGR -> RGB 변환)
    img_cv = cv2.imread(path)
    if img_cv is not None:
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # cv2 실패 시 PIL로 시도
    try:
        pil_img = Image.open(path).convert("RGB")
        arr = np.array(pil_img)
        print(f"PIL로 로드(대체): {path}")
        return arr
    except Exception as e:
        print(f"이미지 읽기 실패(cv2, PIL 모두 실패): {path} -> {e}")
        return None
    
# 1. 세그멘테이션 모델 불러오기
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()
preprocess = weights.transforms()

# 2. 원본 & 배경 불러오기
import os

# 다운로드/저장 디렉토리 (요청하신 경로)
base_path = r"D:/01.project/코드잇/src/TransferModel/synthesis_images"
os.makedirs(base_path, exist_ok=True)

# base_path 내 파일로 경로 설정
img_path = os.path.join(base_path, "test.jpg")
bg_path  = os.path.join(base_path, "background.jpg")

# --- 추가: 이미지가 없으면 다운로드하거나 플레이스홀더 생성 ---
import urllib.request
from PIL import Image as PILImage

def ensure_image(path, url=None, size=(1200, 800)):
    # 이미 존재하면 그대로 사용
    if os.path.exists(path):
        return
    # 1) 먼저 인터넷에서 다운로드 시도 (안정적인 서비스: picsum)
    if url is None:
        url = "https://picsum.photos/{w}/{h}".format(w=size[0], h=size[1])
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp, open(path, 'wb') as out_file:
            out_file.write(resp.read())
        print(f"Downloaded image -> {path}")
        return
    except Exception as e:
        print(f"다운로드 실패 ({url}) : {e}")
    # 2) 다운로드도 실패하면 단색 플레이스홀더 이미지 생성
    try:
        img = PILImage.new("RGB", size, (200, 200, 200))
        img.save(path)
        print(f"Placeholder image created -> {path}")
    except Exception as e:
        print(f"플레이스홀더 생성 실패: {e}")

# test.jpg, background.jpg 확보
ensure_image(img_path, url="https://picsum.photos/1200/800", size=(1200,800))
ensure_image(bg_path,  url="https://picsum.photos/1200/800?random=1", size=(1200,800))

img = load_image_rgb_safe(img_path)
if img is None:
    raise FileNotFoundError(f"원본 이미지를 불러오지 못했습니다: {img_path}")

background = load_image_rgb_safe(bg_path)
if background is None:
    # 배경이 없으면 원본 이미지 복사하거나 단색 배경 생성
    print("배경을 불러오지 못해 원본 이미지를 배경으로 사용합니다.")
    background = img.copy()

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
