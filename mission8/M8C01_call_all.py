import sys
import argparse
from M8C01 import Run_AllModels, Lines

def main():
    """
    M8C01 모듈의 Run_AllModels 함수를 호출하는 스크립트
    모든 모델(UNet, AdvancedUNet, TransferLearningUNet)을 
    모든 Transform 타입(A, B, C, D)으로 학습
    
    사용 예시:
        # 기본값 (에포크 20, 학습률 0.001)
        python M8C01_call_all.py
        
        # 커스텀 에포크 및 학습률
        python M8C01_call_all.py --epochs 30 --lr 0.0005
    """
    parser = argparse.ArgumentParser(description='모든 축구 세그멘테이션 모델 학습')
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=20,
        help='학습 에포크 수 (기본값: 20)'
    )
    
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.001,
        help='학습률 (기본값: 0.001)'
    )
    
    args = parser.parse_args()
    
    Lines(f"전체 모델 학습 시작")
    Lines(f"  - 에포크: {args.epochs}")
    Lines(f"  - 학습률: {args.lr}")
    Lines(f"  - 모델: UNet, AdvancedUNet, TransferLearningUNet")
    Lines(f"  - Transform 타입: A, B, C, D (총 12개 조합)")
    
    # Run_AllModels 호출
    Run_AllModels(
        numEpochs=args.epochs,
        learningRate=args.lr
    )
    
    Lines("=" * 100)
    Lines("전체 모델 학습 완료!")
    Lines("=" * 100)

if __name__ == "__main__":
    main()
