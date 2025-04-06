import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTModel, ViTFeatureExtractor

# -----------------------------
# 1. 参数设置
# -----------------------------
data_dir = "../Aerial_Landscapes"
num_classes = len(os.listdir(data_dir))
batch_size = 16
epochs = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "./vit_checkpoint.pth"  # checkpoint 保存路径

# -----------------------------
# 2. 图像预处理
# -----------------------------
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# -----------------------------
# 3. 数据加载器
# -----------------------------
from torch.utils.data import random_split
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 4. 模型定义：ViT + 线性分类头
# -----------------------------
class ViTForImageClassification(nn.Module):
    def __init__(self, num_classes):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits

# -----------------------------
# 5. 模型训练
# -----------------------------
model = ViTForImageClassification(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(train_loader):.4f}")

    if epoch%4==0:
        # 保存模型 checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(train_loader)
        }
        torch.save(checkpoint, f"./vit_checkpoint_{epoch}.pth" )
        print(f"Checkpoint saved at ./vit_checkpoint_{epoch}.pth")

# -----------------------------
# 6. 模型评估
# -----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
