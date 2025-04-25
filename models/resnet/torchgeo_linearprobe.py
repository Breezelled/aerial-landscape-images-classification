import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import timm
from torchgeo.models import ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(data_dir, target_class)
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
        return image, label

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = '../../data/Aerial_Landscapes'
full_dataset = CustomImageDataset(data_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

class_names = full_dataset.classes
num_classes = len(class_names)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
base_model = timm.create_model("resnet50", pretrained=False, in_chans=3, num_classes=0)
base_model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

model = nn.Sequential(
    base_model,
    nn.Flatten(),
    nn.Linear(2048, num_classes)
)

for param in model[0].parameters():
    param.requires_grad = False

for param in model[1].parameters():
    param.requires_grad = True
for param in model[2].parameters():
    param.requires_grad = True

model = model.to(device)

dummy_input = torch.randn(1, 3, 224, 224).to(device)
try:
    output = model(dummy_input)
    print(f"passed: {output.shape}")
except Exception as e:
    print(f"failed: {str(e)}")
    exit()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': model[1].parameters()},
    {'params': model[2].parameters()}
], lr=0.001)

def calculate_metrics(preds, labels, top_k=(1, 3, 5)):
    metrics = {}
    max_k = max(top_k)
    _, pred_indices = torch.topk(preds, max_k, dim=1)

    for k in top_k:
        correct = pred_indices[:, :k].eq(labels.view(-1, 1))
        correct_k = correct.any(dim=1).float()
        metrics[f'acc@{k}'] = correct_k.mean().item() * 100

    pred_labels = pred_indices[:, 0]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.cpu().numpy(), pred_labels.cpu().numpy(), average='weighted'
    )
    metrics.update({'precision': precision, 'recall': recall, 'f1': f1})
    return metrics


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
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
            outputs = model(inputs)
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
            feature_output = model[0](inputs)
            features.append(feature_output.cpu())
            labels.append(lbls)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()

def visualize_tsne(features, labels, class_names, title="t-SNE Visualization", save_path=None):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(
            features_2d[idx, 0], features_2d[idx, 1],
            c=[colors[i]], label=class_name,
            alpha=0.6
        )

    plt.title(title, fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")
        plt.close()

    plt.show()


def main():
    results_dir = os.path.join("../../results", "torchgeo_resnet50_linearprobe")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

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

    with open(os.path.join(results_dir, "classification_results.txt"), "w") as f:
        f.write("Final Test Results:\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Top-1 Acc: {test_metrics['acc@1']:.2f}% | Top-3 Acc: {test_metrics['acc@3']:.2f}% | Top-5 Acc: {test_metrics['acc@5']:.2f}%\n")
        f.write(f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}\n")
    
    visualize_tsne(
        test_features,
        test_labels,
        class_names=class_names,
        title="t-SNE of TorchGeo Features (Linear Probing)",
        save_path=os.path.join(results_dir, "tsne_visualization.png")
    )


if __name__ == "__main__":
    main()