import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
from torchvision import transforms
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = '../data/Aerial_Landscapes'
output_dir = '../data/split_Aerial_Landscapes'

train_ratio, test_ratio = 0.8, 0.2
for split in ['train', 'test']:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

for category in os.listdir(data_dir):
    category_dir = os.path.join(data_dir, category)
    if not os.path.isdir(category_dir):
        continue

    images = [f for f in os.listdir(category_dir) if f.endswith('.jpg')]
    train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)

    for split, split_images in zip(['train', 'test'], [train_images, test_images]):
        split_category_dir = os.path.join(output_dir, split, category)
        os.makedirs(split_category_dir, exist_ok=True)
        for img in split_images:
            shutil.copy(os.path.join(category_dir, img), os.path.join(split_category_dir, img))


class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir, processor, transform=None):
        self.dataset_dir = dataset_dir
        self.processor = processor
        self.transform = transform
        self.classes = sorted(os.listdir(dataset_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(dataset_dir, target_class)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), label


model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)

num_classes = len(os.listdir(os.path.join(output_dir, 'train')))
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_dataset = CustomImageDataset(os.path.join(output_dir, 'train'), processor, transform)
test_dataset = CustomImageDataset(os.path.join(output_dir, 'test'), processor, transform)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def calculate_metrics(preds, labels, top_k=(1, 3, 5)):
    assert len(preds) == len(labels), "Predictions and labels must have same length"
    labels = labels.to(preds.device).long()

    metrics = {}
    max_k = max(top_k)
    _, pred_indices = torch.topk(preds, max_k, dim=1)

    for k in top_k:
        correct = pred_indices[:, :k].eq(labels.view(-1, 1))
        correct_k = correct.any(dim=1).float()
        metrics[f'acc@{k}'] = correct_k.mean().item() * 100

    pred_labels = pred_indices[:, 0]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.cpu().numpy(),
        pred_labels.cpu().numpy(),
        average='weighted',
        zero_division=0
    )
    metrics.update({'precision': precision, 'recall': recall, 'f1': f1})
    return metrics


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_preds.append(outputs)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_preds, all_labels)
    return running_loss / len(loader), metrics


def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs, output_hidden_states=True)
            feat = outputs.hidden_states[-1]
            features.append(feat.mean(dim=[2, 3]).cpu())
            labels.append(lbls)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def visualize_tsne(features, labels, class_names, title="t-SNE Visualization"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(
            features_2d[idx, 0], features_2d[idx, 1],
            c=colors[i].reshape(1, -1), label=class_name,
            alpha=0.7, s=15
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def main():
    best_f1 = 0.0
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, test_loader, criterion, device)

        print(f"\nEpoch {epoch + 1}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"Top-1 Acc: {val_metrics['acc@1']:.2f}% | Top-3 Acc: {val_metrics['acc@3']:.2f}% | Top-5 Acc: {val_metrics['acc@5']:.2f}%")
        print(
            f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model!")

    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(
        f"Top-1 Acc: {test_metrics['acc@1']:.2f}% | Top-3 Acc: {test_metrics['acc@3']:.2f}% | Top-5 Acc: {test_metrics['acc@5']:.2f}%")
    print(
        f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")

    print("\nGenerating t-SNE visualization...")
    test_features, test_labels = extract_features(model, test_loader)
    visualize_tsne(
        test_features,
        test_labels,
        class_names=test_dataset.classes,
        title="t-SNE of Test Set Features (Best Model)"
    )


if __name__ == "__main__":
    main()
