import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random

input_root = './Aerial_Landscapes'
output_root = './Aerial_Landscapes_Salt'
os.makedirs(output_root, exist_ok=True)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
def add_noise(img_tensor, noise_type='gaussian', std=0.1, sp_ratio=0.01):
    if noise_type == 'gaussian':
        noise = torch.randn_like(img_tensor) * std
        noisy = img_tensor + noise
        return torch.clamp(noisy, 0, 1)

    elif noise_type == 'uniform':
        noise = (torch.rand_like(img_tensor) - 0.5) * 2 * std
        noisy = img_tensor + noise
        return torch.clamp(noisy, 0, 1)

    elif noise_type == 'salt_pepper':
        noisy = img_tensor.clone()
        c, h, w = noisy.shape
        num_pixels = int(h * w * sp_ratio)

        for _ in range(num_pixels):
            i = random.randint(0, h - 1)
            j = random.randint(0, w - 1)
            value = random.choice([0.0, 1.0])
            for ch in range(c):
                noisy[ch, i, j] = value
        return noisy

    else:
        raise ValueError("Unsupported noise type")


def process_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size  # (W, H)

    tensor = to_tensor(img)
    noisy_tensor = add_noise(tensor, noise_type='salt_pepper', std=0.1)

    noisy_img = to_pil(noisy_tensor)
    noisy_img = noisy_img.resize(orig_size, Image.BILINEAR)
    noisy_img.save(output_path)

for root, _, files in os.walk(input_root):
    for fname in files:
        print(fname)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            in_path = os.path.join(root, fname)
            rel_path = os.path.relpath(in_path, input_root)
            out_path = os.path.join(output_root, rel_path)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            process_image(in_path, out_path)
