# 필요한 라이브러리 설치
# pip install pycocotools requests tqdm

import os
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

# COCO 2017 annotations 다운로드 URL
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# 이미지 다운로드 URL prefix
IMG_URL_PREFIX = "http://images.cocodataset.org/train2017/"

BASE_DIR = r"D:\01.project\CodeIt\mission8\data"
# 다운로드한 이미지 저장 폴더
SAVE_DIR = f"{BASE_DIR}/coco_person_dog_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# COCO annotation 파일 로드
coco = COCO(f"{BASE_DIR}/annotations/instances_train2017.json")

# COCO category id 확인
cat_ids = coco.getCatIds(catNms=["person", "dog"])
print("category IDs:", cat_ids)  # person=1, dog=18

# person + dog 둘 다 포함된 이미지 id 찾기
img_ids_person = set(coco.getImgIds(catIds=coco.getCatIds(catNms=["person"])))
img_ids_dog = set(coco.getImgIds(catIds=coco.getCatIds(catNms=["dog"])))

target_img_ids = list(img_ids_person & img_ids_dog)
print("총 이미지 수:", len(target_img_ids))

# 이미지 다운로드
for img_id in tqdm(target_img_ids, desc="Downloading person+dog images"):
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info["file_name"]
    url = IMG_URL_PREFIX + file_name

    save_path = os.path.join(SAVE_DIR, file_name)
    if not os.path.exists(save_path):
        img_data = requests.get(url).content
        with open(save_path, "wb") as f:_
