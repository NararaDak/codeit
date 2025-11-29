import sys
import argparse
from M8C01 import Run_SingleModel,Lines

def main():
    """
    M8C01 모듈의 Run_SingleModel 함수를 호출하는 스크립트
    
    사용 예시:
        # UNet: 기본 U-Net 모델 (가장 빠르지만 성능은 중간)
        python M8C01_call.py --model UNet --epochs 30 --lr 0.001 --transform_type A
        python M8C01_call.py --model UNet --epochs 30 --lr 0.001 --transform_type D
        
        # AdvancedUNet: Attention 메커니즘 적용 (중간 속도, 높은 성능)
        python M8C01_call.py --model AdvancedUNet --epochs 30 --lr 0.001 --transform_type A
        python M8C01_call.py --model AdvancedUNet --epochs 30 --lr 0.001 --transform_type D
        
        # TransferLearningUNet: ResNet34 백본 사용 (느리지만 최고 성능)
        python M8C01_call.py --model TransferLearningUNet --epochs 30 --lr 0.0005 --transform_type A
        python M8C01_call.py --model TransferLearningUNet --epochs 30 --lr 0.0005 --transform_type D
    """
    parser = argparse.ArgumentParser(description='축구 세그멘테이션 모델 학습')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='AdvancedUNet',
        choices=['UNet', 'AdvancedUNet', 'TransferLearningUNet'],
        help='학습할 모델 선택 (기본값: AdvancedUNet)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=30,
        help='학습 에포크 수 (기본값: 30)'
    )
    
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001,
        help='학습률 (기본값: 0.001)'
    )
    
    parser.add_argument(
        '--transform_type',
        type=str,
        default='A',
        choices=['A', 'B', 'C'],
        help='Transform 타입 선택 (기본값: A)'
    )
    
    args = parser.parse_args()
    
    Lines(f"모델 학습 시작")
    Lines(f"  - 모델: {args.model}")
    Lines(f"  - 에포크: {args.epochs}")
    Lines(f"  - 학습률: {args.lr}")
    Lines(f"  - Transform 타입: {args.transform_type}")
    
    # Run_SingleModel 호출
    Run_SingleModel(
        transform_type=args.transform_type,
        modelType=args.model,
        numEpochs=args.epochs,
        learningRate=args.lr
    )
    
    Lines("=" * 100)
    Lines("모델 학습 완료!")
    Lines("=" * 100)

if __name__ == "__main__":
    main()
