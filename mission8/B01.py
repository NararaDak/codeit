import json
import numpy as np
import cv2 as cv

json_path = r"D:\01.project\CodeIt\mission8\data\suup\_annotations.coco.json"

with open(json_path, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]

#넘파이 출력 옵션 세팅 (생략 가능)
np.set_printoptions(threshold=np.inf)

#이미지 1개 선택
img_info = images[0]
h, w = img_info["height"], img_info["width"]
img_id = img_info["id"]

#해당 이미지의 annotation 찾기
anns = [a for a in annotations if a["image_id"] == img_id]

#마스크 초기화
mask = np.zeros((h, w), dtype=np.uint8)


# segmentation -> mask 변환
# segmentation 리스트는 "x, y, x, y, x, y, ..."
# 이렇게 2개씩 한 쌍으로 저장됨으로
# reshape로 x,y 좌표를 찾음
for ann in anns:
    seg = ann["segmentation"][0]
    cat = ann["category_id"]
    poly = np.array(seg).shape
    poly = np.array(seg).reshape(-1, 2)
    print(poly.shape)

    #카테고리 id로 채워줌
    cv.fillPoly(mask, [poly.astype(np.int32)], cat)
    print("-"+str(100))
print(mask.shape)