import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from Understanding_the_Essence.CNN_Visualize.Code.SimpleCNN import SimpleCNN

model = SimpleCNN()
model.eval()

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# 각 합성곱 계층에 hook 등록
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))

# 이미지 로드 및 전처리
img_path = r'C:\junha\Git\Understanding_the_Essence\Understanding_the_Essence\CNN_Visualize\Images\Lenna.png'
image = Image.open(img_path).convert('RGB')

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # 필요시 정규화 과정 추가 (예: transforms.Normalize(mean, std))
])
input_tensor = preprocess(image).unsqueeze(0)  # 배치 차원 추가

# 모델에 이미지 전달 (forward pass)
output = model(input_tensor)

# activation map 시각화 - conv1 (출력 크기: [8, 64, 64])
act1 = activation['conv1'].squeeze(0)  # 배치 차원 제거
num_channels1 = act1.size(0)
plt.figure(figsize=(12, 6))
for i in range(num_channels1):
    plt.subplot(2, 4, i + 1)
    plt.imshow(act1[i].cpu(), cmap='viridis')
    plt.title(f'conv1 - Channel {i}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# activation map 시각화 - conv2 (출력 크기: [16, 32, 32])
act2 = activation['conv2'].squeeze(0)
num_channels2 = act2.size(0)
plt.figure(figsize=(12, 8))
for i in range(num_channels2):
    plt.subplot(4, 4, i + 1)
    plt.imshow(act2[i].cpu(), cmap='viridis')
    plt.title(f'conv2 - Channel {i}')
    plt.axis('off')
plt.tight_layout()
plt.show()
