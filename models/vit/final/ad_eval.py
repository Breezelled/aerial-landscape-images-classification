import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_root = './Aerial_Landscapes'
output_root = './Aerial_Landscapes_Adv'
os.makedirs(output_root, exist_ok=True)

init_mask = 0.001
epsilon = 100
class TrainableBlackDotAttacker(nn.Module):
    def __init__(self, image_shape=(3, 224, 224), epsilon=10):
        super().__init__()
        _, H, W = image_shape
        self.epsilon = epsilon

        self.delta = nn.Parameter(torch.ones(H, W) * init_mask)

    def forward(self, x):
        mask = torch.sigmoid(self.delta * self.epsilon).unsqueeze(0)
        x_adv = x * mask
        return x_adv, self.delta

attacker = TrainableBlackDotAttacker(epsilon=epsilon).to(device)
attacker.load_state_dict(torch.load('./adversarial/attacker_15.pth', map_location=device))
attacker.eval()

resize_transform = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

def process_image(image_path, output_path):
    original_img = Image.open(image_path).convert('RGB')
    orig_size = original_img.size  # (W, H)

    input_tensor = to_tensor(resize_transform(original_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        adv_tensor, _ = attacker(input_tensor)

    adv_image = to_pil(adv_tensor.squeeze(0).clamp(0, 1).cpu())
    adv_image = adv_image.resize(orig_size, Image.BILINEAR)
    adv_image.save(output_path)

for root, _, files in os.walk(input_root):
    for fname in files:
        print(fname)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            in_path = os.path.join(root, fname)
            rel_path = os.path.relpath(in_path, input_root)
            out_path = os.path.join(output_root, rel_path)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            process_image(in_path, out_path)
