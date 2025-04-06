import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification

# === 超参数 ===
num_classes = 15
batch_size = 32
epochs_classifier = 3
epochs_attacker = 3
lr_classifier = 3e-4
lr_attacker = 1e-4
lambda_reg = 10
epsilon = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 数据预处理 ===
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

from torch.utils.data import random_split
data_dir = "../Aerial_Landscapes"
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === 分类器：ViT ===
vit = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes
)
vit = vit.to(device)

optimizer_cls = torch.optim.AdamW(vit.parameters(), lr=lr_classifier)
criterion = nn.CrossEntropyLoss()

# === 微调分类器 ===
vit.train()
for epoch in range(epochs_classifier):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        outputs = vit(pixel_values=x)
        loss = criterion(outputs.logits, y)

        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_cls.step()
    print(f"[Classifier] Epoch {epoch+1}, Loss: {loss.item():.4f}")

# === 微调后评估准确率 ===
vit.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = vit(pixel_values=x)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"微调后准确率: {100 * correct / total:.2f}%")

# === 攻击器定义 ===
class SimpleAttacker(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        delta = self.net(x)
        delta = self.epsilon * delta
        x_adv = torch.clamp(x + delta, 0, 1)
        return x_adv, delta

attacker = SimpleAttacker(epsilon=epsilon).to(device)
optimizer_att = torch.optim.Adam(attacker.parameters(), lr=lr_attacker)

# === 训练攻击器 ===
vit.eval()
for epoch in range(epochs_attacker):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_adv, delta = attacker(x)

        with torch.no_grad():
            outputs = vit(pixel_values=x_adv)
            logits = outputs.logits

        loss_cls = criterion(logits, y)
        loss_reg = delta.abs().sum() / delta.size(0)
        loss = loss_cls + lambda_reg * loss_reg

        optimizer_att.zero_grad()
        loss.backward()
        optimizer_att.step()

    print(f"[Attacker] Epoch {epoch+1}, Loss: {loss.item():.4f}")

# === 评估对抗攻击效果 ===
vit.eval()
correct = 0
total = 0
threshold = 1e-3
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x_adv, delta = attacker(x)
        outputs = vit(pixel_values=x_adv)
        preds = outputs.logits.argmax(dim=1)
        print("标签:", y.tolist())
        print("预测:", preds.tolist())

        changed_pixels = (delta.abs() > threshold).float().sum().item()
        total_pixels = delta.numel()
        print(f"被修改像素数: {int(changed_pixels)} / {total_pixels} ({100 * changed_pixels / total_pixels:.4f}%)")

        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"攻击后准确率: {100 * correct / total:.2f}%")