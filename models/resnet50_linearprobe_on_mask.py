import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import copy

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    best_acc = 0.0
    train_losses = []
    val_losses = []
    best_model = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        y_true_all = []
        y_pred_all = []
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true_all.extend(labels.cpu().numpy())
                y_pred_all.extend(preds.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f} - Val Loss: {val_loss:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
    
    # loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_plot_name = f"loss_curve_{os.path.basename(args.data_dir)}.png"
    plt.savefig(loss_plot_name)
    print(f"📈 Loss curve saved as {loss_plot_name}")

    return best_model, val_loader, class_names

def extract_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)

def top_k_accuracy(model, dataloader, device, k=5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = outputs.topk(k, dim=1)
            correct += sum([labels[i] in pred[i] for i in range(labels.size(0))])
            total += labels.size(0)
    return correct / total

def evaluate_model(model, val_loader, class_names):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []
    features_all = []
    val_running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            features = extract_features(model, inputs)
            outputs = model.fc(features)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
            features_all.append(features.cpu())

    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=class_names))

    features_all = torch.cat(features_all, dim=0).numpy()
    labels = np.array(y_true_all)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb = tsne.fit_transform(features_all)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))

    for i, cls in enumerate(class_names):
        mask = labels == i
        plt.scatter(emb[mask, 0], emb[mask, 1], c=[colors[i]], label=cls, alpha=0.6)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE Visualization")
    plt.grid(True)
    plt.tight_layout()

    tsne_plot_name = f"tsne.png"
    plt.savefig(tsne_plot_name, format="png", bbox_inches='tight')
    plt.close()

    print(f"t-SNE visualization saved at {tsne_plot_name}")

    top3_accuracy = top_k_accuracy(model, val_loader, device, k=3)
    top5_accuracy = top_k_accuracy(model, val_loader, device, k=5)

    report_txt_path = "classification_report.txt"

    report = classification_report(y_true_all, y_pred_all, target_names=class_names)
    accuracy = correct / total

    with open(report_txt_path, "w") as f:
        f.write("Classification Report (Macro):\n")
        f.write(report)
        f.write(f"\nTop-1 Accuracy: {accuracy:.4f}\n")

        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, average='macro'
        )

        f.write(f"Top-3 Accuracy: {top3_accuracy:.4f}\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy:.4f}\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall:    {recall:.4f}\n")
        f.write(f"Macro F1-score:  {f1:.4f}\n")

        print(f"📄 Classification report saved at {report_txt_path}")

def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on image dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/Aerial_Landscapes",
        help="Path to dataset directory (with class subfolders)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate"
    )

    args = parser.parse_args() 
    best_model, val_loader, class_names = train_model(args)
    evaluate_model(best_model, val_loader, class_names)

if __name__ == "__main__":
    main()