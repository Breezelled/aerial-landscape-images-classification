import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# -----------------------------
# 模型定义（与训练时一致）
# -----------------------------
class ViTForImageClassification(nn.Module):
    def __init__(self, num_classes):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            output_attentions=True  # 必须启用注意力输出
        )
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits, outputs.attentions  # 同时返回 logits 和 attentions

# -----------------------------
# 加载模型和 checkpoint
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 15
model = ViTForImageClassification(num_classes=num_classes).to(device)
checkpoint = torch.load("vit_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# -----------------------------
# 加载图像并预处理
# -----------------------------
image_path = "../Aerial_Landscapes/Airport/004.jpg"  # 🔁 替换为你想要可视化的图像路径
image = Image.open(image_path).convert("RGB")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
inputs = processor(images=image, return_tensors="pt").to(device)

# -----------------------------
# 预测类别 & 获取注意力
# -----------------------------
with torch.no_grad():
    logits, attentions = model(pixel_values=inputs['pixel_values'])
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

print(f"✅ Predicted class index: {pred_class}")
print("🔍 Class probabilities:", probs.squeeze().cpu().numpy())

# -----------------------------
# 可视化最后一层 [CLS] 注意力
# -----------------------------
attn = attentions[-1][0]        # shape: [heads, tokens, tokens]
avg_attn = attn.mean(0)         # shape: [tokens, tokens]
cls_attn = avg_attn[0, 1:]      # shape: [196] (CLS -> patches)

# reshape + 归一化 + 上采样
attn_map = cls_attn.reshape(14, 14).cpu().numpy()
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
attn_map_resized = cv2.resize(attn_map, (224, 224))

# 显示热力图
resized_img = image.resize((224, 224))
plt.imshow(resized_img)
plt.imshow(attn_map_resized, cmap='jet', alpha=0.5)
plt.title(f"Predicted Class Index: {pred_class}")
plt.axis('off')
plt.savefig('attention_map.png')
# plt.show()
