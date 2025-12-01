# ------------------------------------------------------------------
# https://huggingface.co/sanz ofr/Stable-diffusion
# ------------------------------------------------------------------

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU 사용 시 float16 제거, GPU 사용 시에만 float16 사용
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# CPU 사용 시 float16 제거, GPU 사용 시에만 float16 사용
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

print("=" * 80)
print("Stable Diffusion 이미지 생성기")
print("종료하려면 Ctrl+C를 누르세요")
print("=" * 80)

try:
    while True:
        prompt = input("\nInput: ").strip()
        
        if not prompt:
            print("프롬프트를 입력해주세요.")
            continue
        
        print(f"이미지 생성 중: '{prompt}'...")
        image = pipe(prompt).images[0]
        
        # 이미지 저장 (프롬프트 기반 파일명)
        filename = f"{prompt[:30].replace(' ', '_')}.png"
        image.save(filename)
        print(f"저장됨: {filename}")
        
        # matplotlib로 이미지 표시
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(prompt)
        plt.tight_layout()
        plt.show()

except KeyboardInterrupt:
    print("\n\n프로그램을 종료합니다.")
except Exception as e:
    print(f"\n오류 발생: {e}")

# for i in range(3):
#     image = pipe(prompt).images[0]
#     image.save(f"astronaut_rides_horse_{i}.png")
#     for i, (imgs, _) in enumerate(dataloader):
#         batch_size = imgs.size(0)
#         real = torch.ones(batch_size, 1).to(device)
#         fake = torch.zeros(batch_size, 1).to(device)

#         # Discriminator 학습
#         optimizer_D.zero_grad()
#         real_imgs = imgs.to(device)
#         output_real = D(real_imgs)
#         loss_real = criterion(output_real, real)

#         z = torch.randn(batch_size, latent_dim).to(device)
#         gen_imgs = G(z)
#         output_fake = D(gen_imgs.detach())
#         loss_fake = criterion(output_fake, fake)

#         loss_D = loss_real + loss_fake
#         loss_D.backward()
#         optimizer_D.step()

#         # Generator 학습
#         optimizer_G.zero_grad()
#         output = D(gen_imgs)
#         loss_G = criterion(output, real)
#         loss_G.backward()
#         optimizer_G.step()
#         if i % 100 == 0:
#             print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] "
#                   f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")