"""
This script generates segmentation masks for a batch of images using the
SAM2 (Segment Anything Model v2) from Facebook Research.

GitHub Repository:
    https://github.com/facebookresearch/sam2.git

Functionality:
- Loads a pre-trained SAM2 model with a specified config and checkpoint.
- Iterates through images in a specified subdirectory.
- Automatically generates segmentation masks for each image.
- Visualizes and saves the mask overlays as output images.

Usage:
1. Install SAM2 according to the instructions in the GitHub repository.
2. Download and place the checkpoint file appropriately.
3. Place input images under: Aerial_Landscapes/<subfolder>/
4. Run the script with the subfolder name as an argument:

    python gen_mask.py <folder_name>
    
"""

import os
import numpy as np
import torch
import torchvision
import sys
from PIL import Image
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

if len(sys.argv) < 2:
    print("用法: python gen_mask.py <子目录名>")
    sys.exit(1)

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


sam2_checkpoint = "/srv/scratch/z5485311/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# 设置路径
input_root = "Aerial_Landscapes"
output_root = "Aerial_Landscapes_Masks"

subfolder = sys.argv[1]  # 运行时传入子目录名

input_subdir = os.path.join(input_root, subfolder)
output_subdir = os.path.join(output_root, subfolder)
os.makedirs(output_subdir, exist_ok=True)

for fname in os.listdir(input_subdir):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    input_path = os.path.join(input_subdir, fname)
    output_path = os.path.join(output_subdir, fname.rsplit(".", 1)[0] + "_mask.jpg")

    # 读取并处理图像
    image = Image.open(input_path).convert("RGB")
    image_np = np.array(image)
    masks = mask_generator.generate(image_np)

    # 可视化并保存
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

print(f"✅ Finished processing: {subfolder}")
