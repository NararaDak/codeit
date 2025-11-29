# YOLOv8n 최적 파라미터 가이드

## 현재 설정
```python
Execute_Training(model_name="YOLOv8n", gubun="partial", epochs=5, lr=0.005, backbone_lr_ratio=0.1, batchSize=4)
```

---

## 📊 YOLOv8n 특성 분석

| 항목 | 특성 |
|------|------|
| **모델 크기** | Nano (초소형, 빠른 추론) |
| **파라미터** | ~3.2M (매우 가볍다) |
| **추천 용도** | 모바일, 엣지 디바이스, 실시간 추론 |
| **학습 속도** | 매우 빠름 |
| **정확도** | 낮음 (크기 대비 필요한 트레이드오프) |

---

## 🎯 추천 파라미터 (3가지 시나리오)

### 1️⃣ **빠른 학습 (Quick Test - 권장 기본설정)**
```python
Execute_Training(
    model_name="YOLOv8n",
    gubun="partial",      # 백본 부분 학습
    epochs=10,            # ↑ 더 많은 에포크 필요
    lr=0.01,              # ↑ YOLOv8은 높은 학습률 권장
    backbone_lr_ratio=0.1,  # 유지
    batchSize=8           # ↑ 배치 크기 증가 (메모리 충분시)
)
```
**특징:**
- 학습 시간: ~5-10분 (GPU 기준)
- 정확도: 중간
- 안정성: 좋음
- **추천 상황:** 프로토타이핑, 테스트

---

### 2️⃣ **균형잡힌 학습 (Balanced - 가장 추천)**
```python
Execute_Training(
    model_name="YOLOv8n",
    gubun="partial",      # 차등 학습률로 백본 살짝 학습
    epochs=20,            # 충분한 에포크
    lr=0.01,              # YOLOv8 표준 값
    backbone_lr_ratio=0.05,  # ↓ 더 낮춤 (백본 보존)
    batchSize=16          # ↑ 더 큰 배치 (안정성)
)
```
**특징:**
- 학습 시간: ~15-20분
- 정확도: 높음
- 안정성: 매우 좋음
- **추천 상황:** 실제 프로젝트, 최고 품질 요구

---

### 3️⃣ **고정확도 학습 (High Accuracy)**
```python
Execute_Training(
    model_name="YOLOv8n",
    gubun="freeze",       # 백본 완전 동결 (전이학습 극대화)
    epochs=30,            # 많은 에포크
    lr=0.001,             # ↓ 낮은 학습률 (세밀한 조정)
    backbone_lr_ratio=0.1,  # 무시됨 (freeze 모드)
    batchSize=32          # ↑ 매우 큰 배치
)
```
**특징:**
- 학습 시간: ~20-30분
- 정확도: 매우 높음
- 안정성: 최고
- **추천 상황:** 세밀한 fine-tuning 필요시, 큰 데이터셋

---

## 🔍 파라미터 상세 설명

### **epochs (에포크)**
| 값 | 특성 | 사용처 |
|-----|------|-------|
| 5-10 | 빠른 학습, 과소 적합 위험 | 빠른 테스트 |
| **15-20** | **균형잡힘, 권장** | **일반적 사용** |
| 30-50 | 느린 학습, 과적합 위험 | 소규모 데이터 |

**YOLOv8n 권장: 15-20 에포크**

---

### **Learning Rate (lr)**
| 값 | 특성 | 사용처 |
|-----|------|-------|
| 0.001 | 매우 낮음, 수렴 느림 | Fine-tuning |
| **0.005-0.01** | **YOLOv8 표준** | **대부분의 경우** |
| 0.05+ | 매우 높음, 불안정 | 피해야 함 |

**YOLOv8n 권장: 0.01**

---

### **backbone_lr_ratio (백본 학습률 비율)**
| 값 | gubun="partial" 적용 | 효과 |
|-----|------|-------|
| 0.1 | 백본 LR = 0.01 × 0.1 = 0.001 | 백본 많이 보존 |
| **0.05** | **백본 LR = 0.01 × 0.05 = 0.0005** | **권장 (세밀함)** |
| 1.0 | 백본과 헤드 같은 LR | 전체 학습 |

**partial 모드에서 권장: 0.05**

---

### **batch size (배치 크기)**
| 값 | 메모리 | 학습 안정성 | 추천 |
|-----|--------|----------|-------|
| 4 | 낮음 | 낮음 | 테스트 |
| 8 | 중간 | 중간 | - |
| **16** | **중간-높음** | **좋음** | **권장** |
| 32+ | 높음 | 매우 좋음 | GPU 충분시 |

**권장: 16 (메모리 충분시 32)**

---

## ✅ 최종 추천 (데이터 크기별)

### **소규모 데이터 (100-500 이미지)**
```python
Execute_Training(
    model_name="YOLOv8n",
    gubun="freeze",
    epochs=20,
    lr=0.001,
    backbone_lr_ratio=0.1,
    batchSize=8
)
```

### **중규모 데이터 (500-2000 이미지) ⭐ 가장 추천**
```python
Execute_Training(
    model_name="YOLOv8n",
    gubun="partial",
    epochs=20,
    lr=0.01,
    backbone_lr_ratio=0.05,
    batchSize=16
)
```

### **대규모 데이터 (2000+ 이미지)**
```python
Execute_Training(
    model_name="YOLOv8n",
    gubun="partial",
    epochs=30,
    lr=0.01,
    backbone_lr_ratio=0.1,
    batchSize=32
)
```

---

## 🚀 GPU 메모리별 추천

| GPU 메모리 | 추천 배치 | 에포크 | 학습 시간 |
|-----------|---------|--------|-----------|
| 2GB | 4-8 | 15 | 30-40분 |
| 4GB | 8-16 | 20 | 20-25분 |
| 8GB+ | 16-32 | 25 | 15-20분 |

---

## ⚡ 학습 모니터링 팁

1. **초기 손실이 안정적인지 확인**
   - 손실이 급격히 변동 → 학습률 감소 필요

2. **검증 mAP 개선 추이 확인**
   - Early Stopping (patience=5)에서 5 에포크 동안 개선 없으면 중단

3. **GPU 메모리 사용량 확인**
   - `nvidia-smi` 명령으로 메모리 모니터링

4. **로그 파일 분석**
   - `op_log.txt`에서 에포크별 손실과 mAP 추이 확인

---

## 🎓 YOLOv8 vs SSD/Faster R-CNN 학습률 비교

| 모델 | 권장 LR | 특징 |
|------|-------|------|
| SSD/Faster R-CNN | 0.001-0.005 | 낮은 LR 선호 |
| **YOLOv8** | **0.01-0.02** | **높은 LR 선호** |
| RetinaNet | 0.001 | 낮은 LR 선호 |

**YOLOv8은 다른 모델보다 높은 학습률에 최적화되어 있습니다!**

---

## 🔧 추가 최적화 팁

### 1. **Warmup 에포크**
```python
# 처음 2-3 에포크는 낮은 LR로 시작
# YOLOv8은 ultralytics API에서 자동 처리
```

### 2. **스케줄러 조정**
```python
# 현재: StepLR(step_size=3, gamma=0.1)
# 10 에포크 후 LR을 10배 감소 권장
# YOLOv8n은 빠르므로 5-7 에포크마다 감소 고려
```

### 3. **데이터 증강**
```python
# YOLOv8은 ultralytics API에서 자동 적용
# 모자이크, 믹스업 등 기본 포함
```

---

## 📈 최종 권장 설정 (현재 코드 수정안)

**현재:**
```python
Execute_Training(model_name="YOLOv8n", gubun="partial", epochs=5, lr=0.005, backbone_lr_ratio=0.1, batchSize=4)
```

**✅ 개선된 권장:**
```python
Execute_Training(model_name="YOLOv8n", gubun="partial", epochs=20, lr=0.01, backbone_lr_ratio=0.05, batchSize=16)
```

**변경 사항:**
- `epochs: 5 → 20` (에포크 4배 증가)
- `lr: 0.005 → 0.01` (학습률 2배 증가 - YOLOv8 최적)
- `backbone_lr_ratio: 0.1 → 0.05` (백본 학습률 더 낮춤)
- `batchSize: 4 → 16` (배치 크기 4배 증가)

**예상 결과:**
- 학습 시간: ~15-20분 (GPU)
- 정확도: 2-3배 향상
- 안정성: 매우 우수

