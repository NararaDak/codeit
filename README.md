# CodeIt Project

코드잇 딥러닝 미션 프로젝트 소스 코드 저장소

## 프로젝트 구조

```
CodeIt/
├── mission7/          # Mission 7: 객체 탐지 (Object Detection)
│   ├── *.py          # Python 소스 코드
│   └── *.ipynb       # Jupyter Notebook 파일
├── mission8/          # Mission 8: 이미지 세그멘테이션 (Image Segmentation)
│   ├── M8C01.py      # 메인 학습 코드
│   ├── M8C01_call.py # CLI 실행 스크립트
│   └── *.ipynb       # Jupyter Notebook 파일
├── back/             # 학습 자료 및 참고 코드
├── doc/              # 문서
└── data/             # 데이터셋 (gitignore에 포함)
```

## Mission 7 - 객체 탐지

- **모델**: SSD300-VGG16, SSDLite-MobileNetV3, YOLOv8n
- **데이터셋**: Oxford-IIIT Pet Dataset
- **주요 기능**: 전이학습, 체크포인트 저장/로드, 평가 지표

## Mission 8 - 이미지 세그멘테이션

- **모델**: UNet, AdvancedUNet (Attention), TransferLearningUNet (ResNet34)
- **데이터셋**: 축구 경기 세그멘테이션
- **주요 기능**: 다양한 데이터 증강, Dice Loss, mIoU 평가

## 환경 설정

```bash
# 가상환경 생성 (프로젝트 루트에서)
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 필요 패키지 설치
pip install torch torchvision
pip install ultralytics  # YOLO
pip install opencv-python pillow
pip install scikit-learn pandas matplotlib
pip install tqdm filelock
```

## 사용법

### Mission 7 - 객체 탐지 테스트

```bash
# Jupyter Notebook에서 실행
cd mission7
jupyter notebook "07_1팀_김영욱 copy 5.ipynb"

# 또는 Python 스크립트로 실행
python E01.py  # YOLO 테스트
python D01.py  # SSD 테스트
```

### Mission 8 - 세그멘테이션 학습

```bash
# CLI를 통한 모델 학습
cd mission8

# UNet 모델 학습
python M8C01_call.py --model UNet --epochs 30 --lr 0.001 --transform_type A

# AdvancedUNet 모델 학습 (Attention 적용)
python M8C01_call.py --model AdvancedUNet --epochs 30 --lr 0.001 --transform_type D

# TransferLearningUNet 모델 학습 (ResNet34 백본)
python M8C01_call.py --model TransferLearningUNet --epochs 30 --lr 0.0005 --transform_type D
```

## 주의사항

- 대용량 데이터 파일(.pth, .pt, 이미지 등)은 `.gitignore`에 포함되어 있습니다.
- 학습 결과는 각 미션의 `modelfiles/` 디렉토리에 저장됩니다.
- 데이터셋은 별도로 다운로드하여 `data/` 디렉토리에 배치해야 합니다.

## 라이선스

Educational Purpose Only
