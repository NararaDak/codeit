import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
import cv2
from transformers import AutoModel,AutoImageProcessor
import os
import torchvision.transforms.functional as TF
import torch.nn.functional as  F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

#────────────────────────────────────────
# 유틸리티 함수
#────────────────────────────────────────
def Lines(text = "", count = 100):
    print("─"*100)
    if text != "":
        print(f"{text}")
        print("─"*count)

#────────────────────────────────────────
# 디렉토리 지정. 
base_dir = "/home/nabi/project/ai/mission/mission5/AI_DATA/Mission5_Data"
base_dir = "D:/AI_DATA/Mission5_Data"
train_dir = f"{base_dir}/train" # Train data X.
train_cleaned_dir = f"{base_dir}/train_cleaned" # Train data Y.(Label)
test_dir = f"{base_dir}/test" # Test data X.
#────────────────────────────────────────
# 변환 클래스. 
#────────────────────────────────────────
class PadToSquare(nn.Module):
    def __init__(self, size: int, fill: float = 0.0):
        super().__init__()
        self.size = size
        self.fill = fill
    
    def forward_org(self, img: torch.Tensor) -> torch.Tensor:
        # 텐서는 (C, H, W) 형태여야 함
        if img.ndim != 3:
            raise ValueError("Input must be a 3D Tensor (C, H, W).")
            
        _, h, w = img.shape
        
        # H, W가 타겟 크기보다 작을 경우에만 패딩 계산
        h_pad = self.size - h
        w_pad = self.size - w
        
        # 패딩은 이미지의 중앙에 오도록 위/아래, 좌/우에 균등하게 분배
        padding_left = w_pad // 2
        padding_right = w_pad - padding_left
        padding_top = h_pad // 2
        padding_bottom = h_pad - padding_top
        
        # 텐서에 패딩 적용 (순서: left, top, right, bottom)
        # fill 값은 텐서의 dtype에 맞춰 float으로 설정해야 합니다.
        return F.pad(img, 
                     [padding_left, padding_top, padding_right, padding_bottom],
                     padding_mode='constant',
                     fill=self.fill)
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # 텐서는 (C, H, W) 형태여야 함
        if img.ndim != 3:
            raise ValueError("Input must be a 3D Tensor (C, H, W).")
            
        _, h, w = img.shape
        
        # H, W가 타겟 크기보다 작을 경우에만 패딩 계산 (여기서는 타겟 크기가 self.size라고 가정)
        h_pad = max(0, self.size - h)
        w_pad = max(0, self.size - w)
        
        # 패딩은 이미지의 중앙에 오도록 위/아래, 좌/우에 균등하게 분배
        padding_left = w_pad // 2
        padding_right = w_pad - padding_left
        padding_top = h_pad // 2
        padding_bottom = h_pad - padding_top
        
        # 텐서에 패딩 적용 (순서: W_left, W_right, H_top, H_bottom)
        # W는 가장 안쪽 차원, H는 그 다음 차원이므로 이 순서가 맞음
        return F.pad(img, 
                     [padding_left, padding_right, padding_top, padding_bottom],
                     mode='constant', # 'padding_mode'를 'mode'로 변경
                     value=self.fill) # 'fill'을 'value'로 변경
        
#────────────────────────────────────────
# 변경 객체
#────────────────────────────────────────
TARGET_SIZE = 224
g_transforms_org = v2.Compose(
    [
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1), 
        # 1. 문서 전체 정보를 보존하며 모델 입력 크기(224x224)에 맞게 축소
        v2.Resize(224, antialias=True), # 420x540이 224x224보다 크므로 축소됨. 가장 짧은 변을 224에 맞춤.
        # 2. 패딩 대신, 축소된 이미지를 224x224 크기에 맞게 패딩 (필요하다면)
        PadToSquare(TARGET_SIZE, fill=1.0),
        # 3. 모델 입력 타입으로 변환 및 정규화
        v2.ToDtype(dtype=torch.float32, scale=True)
    ]
)

g_transforms = transforms.Compose([
    # 1. Resize: 짧은 축을 TARGET_SIZE(224) 이상으로 조정 (필수 전처리)
    # 이미지의 종횡비를 유지하면서 크기를 키워, 다음 단계인 Crop을 준비합니다.
    transforms.Resize(TARGET_SIZE, antialias=True), 
    
    # 2. Crop: 이미지를 TARGET_SIZE x TARGET_SIZE로 강제적으로 잘라냅니다. (크기 통일)
    # 이 단계를 통해 모든 이미지의 [H, W]가 [224, 224]로 고정됩니다.
    transforms.CenterCrop(TARGET_SIZE), 
    
    # 3. ToTensor: PIL 이미지를 텐서 [C, H, W]로 변환 (필수)
    transforms.ToTensor(),
    
    # 4. Normalize: (선택적) 픽셀 값 정규화
    # DINOv2 사전 학습 시 사용된 정규화 값을 사용하는 것이 좋습니다.
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#────────────────────────────────────────
# 이미지 파일 읽기 함수
#────────────────────────────────────────
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):  # PNG 파일만 가져오기
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)  # OpenCV로 이미지를 읽음 (BGR 형식)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
            images.append(img)
    return images
#────────────────────────────────────────
#  X-Y(Label) 데이터를 만든다.(노이즈 이미지와 클린(정답)이미지 생성).
#────────────────────────────────────────
# 이미지 가져 오기 함수.
class loadImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_files = sorted( [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')])
        self.transform = transform
    def __len__(self):
        return len(self.data_files)
    def __getitem__(self, idx):
        img_path = self.data_files[idx]
        image = Image.open(img_path).convert('RGB')
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        return image
# Paired - 이미지(X,y(label) 만들기)
class PairedImageDataset(Dataset):
    def __init__(self, train_dir, train_cleaned_dir, transform=None):
        """
        두 개의 디렉토리에서 이미지를 로드하고 매칭.
        :param train_dir: 원본 이미지 디렉토리
        :param train_cleaned_dir: 정제된 이미지 디렉토리
        :param transform: 원본 이미지에 적용할 데이터 변환
        :param transform_cleaned: 정제된 이미지에 적용할 데이터 변환
        """
        self.train_image = loadImageDataset(train_dir, transform)
        self.cleaned_image = loadImageDataset(train_cleaned_dir, transform)
        assert len(self.train_image) == len(self.cleaned_image), "train과 cleaned 데이터셋의 크기가 다릅니다."
        self.transform = transform
        self.train_len = len(self.train_image)
    def __len__(self):
        return self.train_len
    def __getitem__ort(self, idx):
        return self.train_image, self.cleaned_image
    def __getitem__(self, idx):
        train_img = self.train_image[idx]  # loadImageDataset.__getitem__ 호출 (텐서 반환 예상)
        cleaned_img = self.cleaned_image[idx]
        return train_img, cleaned_img
#────────────────────────────────────────
# 확인용 함수 
#────────────────────────────────────────
def view_paired_dataset(pairedLoader, numImages=5):
    # paired_loader에서 데이터 하나 가져오기
    for trainImages, cleanedImages in pairedLoader:
        # 시각화
        fig, axes = plt.subplots(numImages, 2, figsize=(8, numImages * 3))  # num_images 행, 2열 (위: train, 아래: cleaned)
        for i in range(numImages):
            # 첫 번째 열: 원본 이미지 (train)
            axes[i, 0].imshow(trainImages[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')  # [C, H, W] -> [H, W, C]
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis('off')  # 축 비활성화

            # 두 번째 열: 정제된 이미지 (cleaned)
            axes[i, 1].imshow(cleanedImages[i].permute(1, 2, 0).cpu().numpy(), cmap='gray')  # [C, H, W] -> [H, W, C]
            axes[i, 1].set_title(f"Cleaned {i+1}")
            axes[i, 1].axis('off')  # 축 비활성화

        plt.tight_layout()
        plt.show(block = False)
        plt.pause(3)
        plt.close()
        break  # 한 번만 실행하도록 break
##────────────────────────────────────────
# PairedImageDataset에서 이미지 시각화
#────────────────────────────────────────

#────────────────────────────────────────
# 모델 가져 오기. 
#────────────────────────────────────────

# 모델 정의 함수 (GetExtractAutoMode) 내부 또는 외부에서 모델 구조를 수정해야 합니다.
import torch.nn as nn
# from transformers import AutoModel, AutoImageProcessor # 이미 임포트되었다고 가정

class DenoisingModel(nn.Module):
    def __init__(self, pretrained_backbone, target_image_size=224):
        super().__init__()
        self.backbone = pretrained_backbone
        
        # DINOv2-base의 특징 크기는 768입니다.
        # Vision Transformer의 출력은 [B, N_patches, 768] 형태입니다.
        # 복원을 위해 특징을 이미지와 유사한 2D 형태로 변환하고 업샘플링해야 합니다.
        
        # 임시 복원 헤드: 768차원 특징을 받아 3채널 이미지로 변환 및 업샘플링
        # ⚠️ 이 부분은 DINOv2의 패치 크기와 이미지 크기에 따라 복잡한 Reshape 및 Transpose 과정이 필요합니다.
        # 단순화를 위해 ConvTranspose 레이어 시퀀스로 예시를 듭니다.
        self.restoration_head = nn.Sequential(
           # 1. 16x16 -> 32x32 (입력 768)
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 2. 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 3. 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # 🟢 마지막 층: 128x128 -> 224x224로 변환해야 합니다.
            # 가장 확실한 방법은 Stride=1, Kernel=1을 사용하여 채널만 조정한 후, 
            # F.interpolate를 사용하여 원하는 크기로 업샘플링하는 것입니다.
            # 여기서는 ConvTranspose로만 해결하겠습니다.
            
            # 4. 128x128 -> 224x224 (출력 크기 224를 강제하는 복잡한 설정)
            # 128에서 224가 되려면 확장 인자가 1.75여야 합니다. 
            # ConvTranspose로 1.75배 확장은 불가능하므로, 
            # 마지막 레이어 전에 크기를 $112 \times 112$로 낮춰야 합니다.
            
            # 3번 층의 출력을 112로 강제하는 것이 더 쉽습니다.
            # 64x64 -> 112x112: K=3, S=2, P=1, OP=0 (Output = 2*64 - 2*1 + 3 = 129 -> 안됨)
            
            # 128x128에서 224x224로 가는 ConvTranspose (Stride=1, Kernel=1을 제외한)는 너무 복잡합니다.
            
            # 🌟 대안: 128x128에서 Stride 1의 Conv를 통과시킨 후, F.interpolate를 사용합니다. 🌟
            nn.Conv2d(64, 3, kernel_size=3, padding=1), # 채널만 3으로 조정 (크기 128x128 유지)
            nn.Upsample(size=(target_image_size, target_image_size), mode='bilinear', align_corners=False), # 224x224로 강제 조정
            #nn.Sigmoid()
        )
        
        self.target_size = target_image_size
        
        # 🟢 수정된 부분: patch_size를 num_patches 계산 전에 정의해야 합니다.
        self.patch_size = 16 # DINOv2-base의 기본 패치 크기
        # 패치 토큰 수에서 클래스 토큰 제외 등을 처리하기 위한 임시 차원 변환기
        self.target_size = target_image_size
        self.num_patches = (self.target_size // self.patch_size) ** 2

    def forward(self, x):
        # 1. 특징 추출: outputs.last_hidden_state는 [B, N_patches+1, 768] (클래스 토큰 포함)
        features = self.backbone(x).last_hidden_state
        
        # 2. 클래스 토큰 제외 (N_patches = 196을 얻기 위해)
        patch_features = features[:, 1:, :] # [B, 196, 768]
        
        # 3. 1D 특징을 2D 그리드로 Reshape
        B, N, C = patch_features.shape
        H = W = int(N**0.5) # H=W=14
        
        # [B, N_patches, 768] -> [B, 768, 14, 14]
        patch_features = patch_features.permute(0, 2, 1).view(B, C, H, W)
        
        # 4. 복원 헤드를 통과 (14x14 -> 224x224)
        output_image = self.restoration_head(patch_features)
        return torch.sigmoid(output_image)

def GetExtractAutoMode():
    modelName = "facebook/dinov2-base"
    image_processor = AutoImageProcessor.from_pretrained(modelName)
    Lines()
    print(f"feature:{image_processor}")
    Lines()
    pretrainedModel = AutoModel.from_pretrained(modelName)
    return DenoisingModel(pretrainedModel)

#────────────────────────────────────────
# 추가 훈련 함수. 
#────────────────────────────────────────
g_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 장치 설정
def ExtraTrain(pairedLoader, epochs, lr):
   
    pretrainedModel = GetExtractAutoMode().to(g_Device) # 모델을 장치로 이동
    # 1. 손실 함수 정의 (MSE Loss)
    loss_fn = nn.MSELoss() 
    # 2. 옵티마이저 정의 (Adam)
    optimizer = optim.Adam(pretrainedModel.parameters(), lr=lr)
    pretrainedModel.train()
    #────────────────────────────────────────
    # 🌟 디버깅 코드 추가 🌟
    #────────────────────────────────────────
    # DataLoader에서 첫 번째 배치 데이터 가져오기
    with torch.no_grad():
        for trainImages_test, cleanImages_test in pairedLoader:
            trainImages_test = trainImages_test.to(g_Device)
            outputs_test = pretrainedModel(trainImages_test)
            Lines("DEBUG: OUTPUT SHAPE CHECK")
            print(f"Model Output Shape (outputs): {outputs_test.shape}")
            print(f"Target Image Shape (cleanedImages, expected): {trainImages_test.shape}")
            Lines()
            
            cleanImages_test = cleanImages_test.to(g_Device)
            outputs_test = pretrainedModel(cleanImages_test)
            Lines("DEBUG: OUTPUT SHAPE CHECK")
            print(f"Model Output Shape (outputs): {outputs_test.shape}")
            print(f"Cleaned Image Shape (cleanedImages, expected): {cleanImages_test.shape}")
            Lines()
            break # 첫 배치만 확인
    #────────────────────────────────────────
    # 🌟 디버깅 코드 종료 🌟
    #────────────────────────────────────────
 
    # 학습 시작
    for epoch in range(epochs):
        total_loss = 0
        # for trainImages, cleanedImages in tqdm(pairedDataset, desc=f"Epoch {epoch+1}"): # DataLoader가 아닌 Dataset을 직접 순회
        index = 0
        for trainImages, cleanedImages in pairedLoader:
            # 데이터를 장치로 이동
            trainImages = trainImages.to(g_Device)
            cleanedImages = cleanedImages.to(g_Device)

            # 순전파
            outputs = pretrainedModel(trainImages)
            
            # 손실 계산
            loss = loss_fn(outputs, cleanedImages)
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"[{index}/{len(pairedLoader)}], Loss: {loss.item():.4f}", end='\r')
            index += 1

        avg_loss = total_loss / len(pairedLoader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    print("Fine-tuning complete.")
    return pretrainedModel # 훈련된 모델 반환


#EPOCHS = 5 
#LEARN_RATE = 0.00001
def Execute_Model(EPOCHS,LEARN_RATE):
    #────────────────────────────────────────
    # 이미지 불러 오기.
    #────────────────────────────────────────
    g_train_images = load_images_from_folder(train_dir)
    g_train_cleaned_images = load_images_from_folder(train_cleaned_dir)
    g_test_images = load_images_from_folder(test_dir)
    print(f"Train: {len(g_train_images)}")
    print(f"Train Cleaned: {len(g_train_cleaned_images)}")
    print(f"Test: {len(g_test_images)}")
    Lines("Create Data loader. ")
    # 데이터셋 생성
    pairedDataset = PairedImageDataset(train_dir, train_cleaned_dir, g_transforms)
    #test_dataset = loadImageDataset(test_dir, g_transforms) 
    #────────────────────────────────────────
    # PairedImageDataset에서 이미지 시각화
    #───────────────────────────────────────
    pairedLoader = DataLoader(pairedDataset, batch_size=16, shuffle=True, num_workers=0)
    view_paired_dataset(pairedLoader, numImages=5)
    #────────────────────────────────────────
    # 훈련 
    #───────────────────────────────────────
    Lines("모델 훈련을 시작합니다.")
    return ExtraTrain(pairedLoader, EPOCHS, LEARN_RATE)
    

def evalModel(model,className,epoches,learnRate):    
# 테스트
    model.eval()
    testDataset = loadImageDataset(test_dir, g_transforms) 
    testLoader = DataLoader(testDataset, batch_size=16, shuffle=False, num_workers = 0)
    with torch.no_grad():
        nIndex = 1
        for images in testLoader:
            images_batch = images.to(g_Device) # 원본 이미지
            outputs_batch = model(images_batch) # 출력
            saveResultFile(nIndex, images_batch, outputs_batch,className,epoches,learnRate)
            visualize_images_and_outputs(images, outputs_batch)
            nIndex +=1
    Lines("View images/outputs")
def visualize_images_and_outputs(images, outputs):
    """
    이미지와 출력 이미지를 열로 구분하여 시각화.
    :param images: 원본 이미지 텐서
    :param outputs: 모델 출력 텐서
    """
    num_images = images.size(0)  # 전체 이미지 개수
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))  # num_images 행, 2열

    for i in range(num_images):
        # 첫 번째 열: 원본 이미지
  #     axes[i, 0].imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i, 0].imshow(images[i].cpu().numpy().squeeze().transpose((1, 2, 0)))
        axes[i, 0].set_title(f"Original {i + 1}", fontsize=10)
        axes[i, 0].axis('off')

        # 두 번째 열: 출력 이미지
        #axes[i, 1].imshow(outputs[i].cpu().detach().numpy().squeeze(), cmap='gray')
        axes[i, 1].imshow(outputs[i].cpu().numpy().squeeze().transpose((1, 2, 0)))
        axes[i, 1].set_title(f"Output {i + 1}", fontsize=10)
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show(block = False)
    plt.pause(3)
    plt.close()

def saveResultFile(nIndex, images, outputs, className, epoches, learnRate):
    """
    원본 이미지와 출력 이미지를 하나의 파일로 합쳐서 저장합니다.
    """
    # 텐서를 CPU로 이동하고 NumPy 배열로 변환
    # detach()는 outputs에만 적용되어야 합니다. images는 보통 require_grad=False이므로 괜찮지만, 명확하게 분리합니다.
    images_np = images.cpu().numpy()
    outputs_np = outputs.cpu().detach().numpy()

    # 배치의 각 이미지에 대해 반복
    for i in range(images_np.shape[0]):
        # 0-1 범위를 0-255 범위로 변환하고 uint8 타입으로 변경
        
        # 1. squeeze() 적용 (불필요한 배치 차원 제거)
        img_temp = images_np[i].squeeze()
        out_temp = outputs_np[i].squeeze()
        
        # 2. 채널 순서 변경 (C, H, W -> H, W, C)
        # 만약 이미지가 흑백(H, W)이라면 transpose는 필요 없으며 오류가 발생할 수 있습니다.
        # 따라서 차원이 3개인 경우(컬러 이미지)에만 transpose를 적용합니다.
        if img_temp.ndim == 3: # 컬러 이미지 (C, H, W)인 경우
            img_temp = img_temp.transpose((1, 2, 0))
            out_temp = out_temp.transpose((1, 2, 0))
            
        # 3. 0-255 범위로 변환 및 uint8 타입으로 변경
        img = (img_temp * 255).astype(np.uint8)
        out = (out_temp * 255).astype(np.uint8)

        # 원본과 출력을 수평으로 연결
        # 두 배열의 (높이, 채널)이 같아야 hstack이 가능합니다.
        combined_img = np.hstack((img, out))
        
        # 저장 경로 설정
        imagedir = f"{base_dir}/model_result_image"
        save_dir = f"{imagedir}/{className}_{epoches}_{learnRate}"
        os.makedirs(save_dir, exist_ok=True) 

        # 파일 이름 설정 (배치 인덱스_이미지 인덱스)
        file_name = f"{save_dir}/result_{nIndex-1}_{i}.png"

        print(f"out shape: {out.shape}, out dtype: {out.dtype}, out min: {out.min()}, out max: {out.max()}")        
        # PIL을 사용하여 저장: 이제 combined_img의 형태는 (Height, Width, Channels)가 됩니다.
        Image.fromarray(combined_img).save(file_name)
MODEL_CLASS_NAME  = "DenoisingModel"
EPOCHES = 5
LEARN_RATE = 0.00001

MODEL_PATH = f"{base_dir}/modelfiles/{MODEL_CLASS_NAME}_{EPOCHES}_{LEARN_RATE}.pth"

def GetModel():
    # 모델 파일이 존재하면 로드, 없으면 훈련
    if os.path.exists(MODEL_PATH):
        print(f"'{MODEL_PATH}' 파일이 존재하여, 저장된 모델을 불러옵니다.")
        # 저장된 모델 전체를 로드합니다.
        model = torch.load(MODEL_PATH, map_location=g_Device, weights_only=False)
        return model
    else:
        print(f"'{MODEL_PATH}' 파일이 없어, 모델 훈련을 시작합니다.")
        # 모델 훈련
        model = Execute_Model(EPOCHES,LEARN_RATE)
        # 훈련된 모델 저장
        torch.save(model, MODEL_PATH)
        print(f"훈련된 모델을 '{MODEL_PATH}' 파일로 저장했습니다.")
        return model
model_class = GetModel()
evalModel(model_class,MODEL_CLASS_NAME,EPOCHES,LEARN_RATE)
