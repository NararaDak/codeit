import torch as tc
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from transformers import pipeline


print(tc.__version__)
print(tc.cuda.is_available())


#────────────────────────────────────────
# 유틸리티 함수
#────────────────────────────────────────
def Lines(linecount=90):
    print("─" * linecount + "\n")
#────────────────────────────────────────
def test_test():
    # 텍스트 감정 분석을 위한 파이프라인 객체를 생성합니다.
    # 이 과정에서 사전 학습 모델이 자동으로 다운로드됩니다.
    classifier = pipeline("sentiment-analysis")

    # 간단한 문장을 넣어 모델을 테스트합니다.
    result = classifier("I love using Hugging Face transformers!")

    print(result)


#────────────────────────────────────────
# 사전 학습된 ResNet18 모델 불러오기
#────────────────────────────────────────
model = models.resnet18(weights="IMAGENET1K_V1")
print(model)
Lines()

