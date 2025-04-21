import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTModel,ViTFeatureExtractor
from dataset import get_data_loader

# Hyper parameters
trained_epochs = 20
epochs = 5
lr_classifier = 4e-4
# 8e-3,8e-2
lr_attacker = 2e-2
lambda_reg = 8
#1,0.5,0.5,0.5,0.5,
# 如果不能误导，则太局限，lambda_reg要改小
# 如果不能减少黑块，则太松弛，lambda_reg要增大
init_mask = 0.001
epsilon = 100
# init_mask会随机影响初始攻击效果
# epsilon

image_dir= "../../../../../Downloads/Aerial_Landscapes"
num_classes = 15
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ViTForImageClassification(nn.Module):
    def __init__(self, num_classes):
        super(ViTForImageClassification, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_size = 224
        
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits
        
vit = ViTForImageClassification(num_classes=num_classes).to(device)
checkpoint = torch.load("./save/ViT_all/15.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
image_size = vit.image_size
image_mean = vit.feature_extractor.image_mean
image_std = vit.feature_extractor.image_std

# Load Dataset
train_loader, val_loader, test_loader = get_data_loader(image_dir=image_dir,
                                                        image_size=image_size,
                                                        expected_mean=image_mean,
                                                        expected_std=image_std,
                                                        batch_size=batch_size)

class TrainableBlackDotAttacker(nn.Module):
    def __init__(self, image_shape=(3, 224, 224), epsilon=10):
        super().__init__()
        _, H, W = image_shape
        self.epsilon = epsilon
        
        self.delta = nn.Parameter(torch.ones(H, W)*init_mask)

    def forward(self, x):
        mask = torch.sigmoid(self.delta * self.epsilon).unsqueeze(0)
        x_adv = x * mask
        return x_adv, self.delta

attacker = TrainableBlackDotAttacker(epsilon=epsilon).to(device)
if trained_epochs:
    print("load state dict")
    attacker.load_state_dict(torch.load(f'./adversarial/attacker_{trained_epochs}.pth', map_location=torch.device('cpu')))

# Loss and Optimizer
criterion_cls = nn.CrossEntropyLoss()
criterion_att = nn.CrossEntropyLoss()
optimizer_cls = torch.optim.AdamW(vit.parameters(), lr=lr_classifier)
optimizer_att = torch.optim.Adam(attacker.parameters(), lr=lr_attacker)
accuracy_list = []
loss_list = []

# Evaluate at Begin
vit.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y, _ in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = vit(x)
        
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"Accuracy at begin: {100 * correct / total:.2f}%")

# Mkdir
dir_dir = os.path.dirname(f"./adversarial/")
os.makedirs(dir_dir, exist_ok=True)

# Train
for epoch in range(trained_epochs,trained_epochs+epochs):
    # lambda_reg = lambda_reg_list[epoch]

    if epoch>=0: # epooch%2==1 and epoch>0:
        sum_loss=0
        vit.eval()
        attacker.train()
        for i, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x_adv, delta = attacker(x)
            outputs = vit(x_adv)
    
            # regularization
            loss_reg = -torch.sigmoid(delta).sum() / delta.size(0) / delta.size(1)
            loss_miss = -criterion_att(outputs, y)
            if i%10==0:
                print(i,len(train_loader))
                print(loss_miss, loss_reg)
                print(torch.sigmoid(delta))

            total_loss =  lambda_reg * loss_reg + loss_miss
            sum_loss+=total_loss
            optimizer_att.zero_grad()
            total_loss.backward()
            optimizer_att.step()
            
        print(f"[Epoch {epoch + 1}] Loss: {sum_loss/len(train_loader):.4f}")
        loss_list.append(loss_reg)

        # Evaluate at Mid
        vit.eval()
        attacker.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in test_loader:
                x, y = x.to(device), y.to(device)
                x_adv, delta = attacker(x)
                outputs = vit(x_adv)
                
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Accuracy after {epoch+1} attach: {100 * correct / total:.2f}%")
        accuracy_list.append(100 * correct / total)

    if (epoch+1) %5 == 0:
        torch.save(attacker.state_dict(), f'./adversarial/attacker_{epoch+1}.pth')

# Evaluate at end
vit.eval()
attacker.eval()
correct = 0
total = 0
threshold = 1e-10
with torch.no_grad():
    for x, y, _ in test_loader:
        x, y = x.to(device), y.to(device)
        x_adv, delta = attacker(x)
        outputs = vit(pixel_values=x_adv)
        preds = outputs.argmax(dim=1)

        ## result
        # print("label:", y.tolist())
        # print("predict:", preds.tolist())

        # changed_pixels = (delta.abs() > threshold).float().sum().item()
        # total_pixels = delta.numel()
        # print(f"{int(changed_pixels)} / {total_pixels} ({100 * changed_pixels / total_pixels:.4f}%)")

        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Final Accuracy: {100 * correct / total:.2f}%")

print(accuracy_list)
print(loss_list)

# accuracy_list = [t.item() for t in accuracy_list]
loss_list = [t.item() for t in loss_list]

import pandas as pd
loss_df = pd.DataFrame({
    'accuracy_list': accuracy_list,
    'loss_list': loss_list
})

loss_df.to_csv(f"./adversarial/{trained_epochs+1}to{trained_epochs+epochs}_accuracy.csv", index=False)


