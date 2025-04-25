import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

epsilon = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainableBlackDotAttacker(nn.Module):
    def __init__(self, image_shape=(3, 224, 224), epsilon=10):
        super().__init__()
        _, H, W = image_shape
        self.epsilon = epsilon
        self.delta = nn.Parameter(torch.zeros(H, W))

    def forward(self, x):
        mask = torch.sigmoid(self.delta * self.epsilon).unsqueeze(0)
        x_adv = x * mask
        return x_adv, self.delta

attacker = TrainableBlackDotAttacker(epsilon=epsilon).to(device)
attacker.load_state_dict(torch.load('save/attacker_25.pth', map_location=device))
attacker.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open('./Aerial_Landscapes/Beach/009.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

with torch.no_grad():
    adv_img_tensor, _ = attacker(img_tensor)

# === 噪声函数 ===
def add_black_white_salt_pepper_noise(img_tensor, amount=0.02):
    noisy = img_tensor.clone()
    _, c, h, w = noisy.shape
    num_pixels = int(h * w * amount)

    for _ in range(num_pixels):
        i = np.random.randint(0, h)
        j = np.random.randint(0, w)
        value = float(np.random.choice([0.0, 1.0]))  # 黑或白
        noisy[0, :, i, j] = value  # 所有通道统一赋值
    return noisy

def add_gaussian_noise(img_tensor, mean=0.0, std=0.1):
    noise = torch.randn_like(img_tensor) * std + mean
    return (img_tensor + noise).clamp(0, 1)

sp_tensor = add_black_white_salt_pepper_noise(img_tensor)
gauss_tensor = add_gaussian_noise(img_tensor)

# how many pixels have been changed
threshold = 0.01
changed_pixels = (abs(adv_img_tensor - img_tensor) > threshold).float().sum().item()
total_pixels = img_tensor.numel()
print(f"{int(changed_pixels)} / {total_pixels} ({100 * changed_pixels / total_pixels:.4f}%)")

# show images
def imshow(tensor, title=''):
    img = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
imshow(img_tensor, 'Original Image')

plt.subplot(2, 1, 2)
imshow(adv_img_tensor, 'Adversarial Image')

# plt.subplot(2, 2, 3)
# imshow(sp_tensor, 'Salt & Pepper Noise')
#
# plt.subplot(2, 2, 4)
# imshow(gauss_tensor, 'Gaussian Noise')

plt.tight_layout()
plt.show()
