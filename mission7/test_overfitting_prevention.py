"""
테스트 스크립트: 과적합 방지 기법 검증
- epochs=3으로 짧게 학습
- Early Stopping, Gradient Clipping 동작 확인
- 학습/검증 손실 및 mAP 추적
"""
import os
import sys
sys.path.insert(0, r"d:\05.gdrive\codeit\mission7\src")

from A04 import (
    MyMeta, GetLoader, get_default_transforms,
    SSDLiteMobileNetV3Transfer, OpLog, Lines
)

def main():
    Lines("=== 과적합 방지 기법 테스트 시작 ===")
    
    meta = MyMeta()
    
    # 데이터 로더 생성 (배치 크기를 작게 설정해 빠른 테스트)
    train_list, val_list = meta.trainval_list[:100], meta.trainval_list[100:150]  # 매우 작은 샘플
    
    # Transform 설정
    transforms = get_default_transforms()
    train_loader, val_loader, test_loader = GetLoader(meta, train_list, val_list, meta.test_list[:50], transforms, batchSize=8)
    
    # 모델 생성 (epochs=3, 짧은 학습률)
    model = SSDLiteMobileNetV3Transfer(meta=meta, gubun="partial", epochs=3, lr=0.0001, backbone_lr_ratio=0.1)
    
    Lines(f"Model: {model.getMyName()}")
    Lines(f"Train samples: {len(train_list)}, Val samples: {len(val_list)}")
    Lines(f"Epochs: 3, LR: 0.0001, Backbone LR Ratio: 0.1")
    
    # 학습 시작
    try:
        model.train(train_loader, val_loader, test_loader)
        Lines("✓ 학습 완료")
    except Exception as e:
        Lines(f"✗ 학습 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
