import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader, random_split

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import argparse
from pathlib import Path
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"


class AerialImageDataset(Dataset):
    def __init__(self, dataset_dir, processor, image_paths=None, labels=None, class_names=None):
        self.processor = processor
        self.image_paths = []
        self.labels = []

        if image_paths is not None and labels is not None and class_names is not None:
            # Initializing from pre-split data
            self.image_paths = image_paths
            self.labels = labels
            self.class_names = class_names
        else:
            # Initializing from directory (for the first time)
            self.class_names = sorted(os.listdir(dataset_dir))
            class_to_idx = {name: i for i, name in enumerate(self.class_names)}

            for class_name in self.class_names:
                folder = os.path.join(dataset_dir, class_name)
                cls_idx = class_to_idx[class_name]
                for fname in os.listdir(folder):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    path = os.path.join(folder, fname)
                    self.image_paths.append(path)
                    self.labels.append(cls_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            # Remove batch dimension from HF processor
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy item or handle appropriately
            # For simplicity, we might skip this item in collate_fn or raise error
            # Here, let's return None and handle in collate_fn
            return None


        return {
            "inputs": inputs,
            "label": self.labels[idx],
            "path": image_path,
        }

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class CLIPLinearProbeLogitWrapper(nn.Module):
    def __init__(self, clip_model, linear_probe):
        super().__init__()
        self.clip_model = clip_model
        self.linear_probe = linear_probe

    def forward(self, pixel_values):
        """
        pixel_values: shape (B, 3, H, W)
        Returns logits from the linear probe. Shape (B, num_classes)
        """
        # Extract features using the frozen CLIP model
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            # No normalization needed before linear layer usually,
            # but CLIP features are often normalized. Let's keep it consistent.
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Pass features through the trainable linear probe
        logits = self.linear_probe(image_features)
        return logits


def collate_fn(batch):
    # Filter out None items caused by loading errors
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch failed

    input_keys = batch[0]["inputs"].keys()
    batch_inputs = {
        k: torch.stack([item["inputs"][k] for item in batch]) for k in input_keys
    }
    batch_labels = [item["label"] for item in batch]
    batch_paths = [item["path"] for item in batch]
    return {
        "inputs": batch_inputs,
        "labels": torch.tensor(batch_labels),
        "paths": batch_paths,
    }


# --- Main Function for Linear Probing ---
def run_clip_linear_probing(
    dataset_dir="../data/Aerial_Landscapes",
    save_dir="../results/clip_linear_probe",
    batch_size=32,
    model_path=None,
    epochs=10,
    lr=1e-3,
    train_split_ratio=0.8,
):
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Load CLIP Model (Feature Extractor) ---
    if model_path and os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(f"Loading base CLIP model from {model_path}")
        clip_model = CLIPModel.from_pretrained(model_path).to(device)
    elif model_path and os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
         print(f"Loading base CLIP model from {model_path}")
         clip_model = CLIPModel.from_pretrained(model_path).to(device)
    else:
        print(f"Loading pre-trained CLIP model {model_name}")
        clip_model = CLIPModel.from_pretrained(model_name).to(device)

    processor = CLIPProcessor.from_pretrained(model_name)

    # --- Freeze CLIP Model Parameters ---
    print("Freezing CLIP model parameters...")
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval() # Set CLIP to evaluation mode

    # --- 2. Prepare Dataset and Dataloaders ---
    full_dataset = AerialImageDataset(dataset_dir, processor)
    class_names = full_dataset.class_names
    num_classes = len(class_names)

    total_size = len(full_dataset)
    train_size = int(total_size * train_split_ratio)
    val_size = total_size - train_size # Use the remaining part for validation

    print(f"Total dataset size: {total_size}, Training size: {train_size}, Validation size: {val_size}")

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # --- 3. Define Linear Classifier, Loss, Optimizer ---
    image_feature_dim = clip_model.visual_projection.out_features
    linear_probe = LinearProbe(image_feature_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_probe.parameters(), lr=lr)

    # --- 4. Training Loop ---
    best_val_acc = 0.0
    best_model_state = None

    print("\n--- Starting Linear Probe Training ---")
    for epoch in range(epochs):
        linear_probe.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            if batch is None: continue # Skip bad batches
            batch_inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            batch_labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Extract features (no gradients needed for CLIP)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**batch_inputs)
                # Normalize features - consistent with CLIP's typical use
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            # Forward pass through linear probe
            logits = linear_probe(image_features)
            loss = criterion(logits, batch_labels)

            # Backward pass and optimize *only* linear probe
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_labels.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            pbar.set_postfix(loss=loss.item(), acc=train_correct/train_total if train_total > 0 else 0)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = train_correct / train_total if train_total > 0 else 0
        print(f"Epoch {epoch+1} Train Loss: {epoch_loss}, Train Acc: {epoch_acc}")

        # --- Validation Step ---
        linear_probe.eval() # Set linear probe to evaluation mode
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in pbar_val:
                if batch is None: continue
                batch_inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
                batch_labels = batch["labels"].to(device)

                image_features = clip_model.get_image_features(**batch_inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                logits = linear_probe(image_features)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item() * batch_labels.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

                all_val_labels.extend(batch_labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                pbar_val.set_postfix(acc=val_correct/val_total if val_total > 0 else 0)


        val_epoch_acc = val_correct / val_total if val_total > 0 else 0
        val_epoch_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch+1} Val Loss: {val_epoch_loss}, Val Acc: {val_epoch_acc}")

        # Save best model based on validation accuracy
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_state = copy.deepcopy(linear_probe.state_dict())
            print(f"*** New best validation accuracy: {best_val_acc} ***")
            # Save the best linear probe weights
            torch.save(best_model_state, os.path.join(save_dir, "best_linear_probe.pth"))

        # Optional: LR Scheduler step
        # scheduler.step()

    print("--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_acc}")

    # Load best model state for final evaluation
    if best_model_state:
        linear_probe.load_state_dict(best_model_state)
        print("Loaded best linear probe model for final evaluation.")
    else:
        print("Warning: No best model state saved. Using the last epoch's model.")

    # --- 5. Final Evaluation on Validation Set ---
    linear_probe.eval()
    true_labels = []
    pred_labels = [] 
    all_image_features = [] # For t-SNE
    all_image_labels = []   # For t-SNE

    top1_count = top3_count = top5_count = 0

    print("\n--- Final Evaluation on Validation Set ---")
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
             if batch is None: continue
             batch_inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
             batch_labels_cpu = batch["labels"]

             image_features = clip_model.get_image_features(**batch_inputs)
             image_features_normalized = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
             logits = linear_probe(image_features_normalized)

             topk_values, topk_indices = torch.topk(logits, k=5, dim=-1)
             topk_indices_cpu = topk_indices.cpu().numpy()

             for i in range(batch_labels_cpu.shape[0]):
                 true_label = batch_labels_cpu[i].item()
                 top5_preds = topk_indices_cpu[i]

                 if true_label == top5_preds[0]:
                     top1_count += 1
                 if true_label in top5_preds[:3]:
                     top3_count += 1
                 if true_label in top5_preds:
                     top5_count += 1

             predictions = torch.argmax(logits, dim=-1).cpu()
             true_labels.extend(batch_labels_cpu.numpy())
             pred_labels.extend(predictions.numpy())

             all_image_features.append(image_features.cpu().numpy())
             all_image_labels.extend(batch_labels_cpu.numpy())


    all_image_features = np.concatenate(all_image_features, axis=0)
    all_image_labels = np.array(all_image_labels)

    print("Classification Report (Validation Set):")
    report = classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0)
    print(report)

    total_samples = len(val_dataset)
    acc1 = top1_count / total_samples
    acc3 = top3_count / total_samples
    acc5 = top5_count / total_samples

    macro_precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    print(f"\nValidation Top-1 Accuracy: {acc1}")
    print(f"Validation Top-3 Accuracy: {acc3}")
    print(f"Validation Top-5 Accuracy: {acc5}")
    print(f"Macro Precision: {macro_precision}")
    print(f"Macro Recall:    {macro_recall}")
    print(f"Macro F1-score:  {macro_f1}")

    result_txt_path = os.path.join(save_dir, "classification_results.txt")
    with open(result_txt_path, "w") as f:
        f.write("Classification Report (Validation Set):\n")
        f.write(report)
        f.write(f"\nValidation Top-1 Accuracy: {acc1}\n")
        f.write(f"Validation Top-3 Accuracy: {acc3}\n")
        f.write(f"Validation Top-5 Accuracy: {acc5}\n")
        f.write(f"Macro Precision: {macro_precision}\n")
        f.write(f"Macro Recall:    {macro_recall}\n")
        f.write(f"Macro F1-score:  {macro_f1}\n")
        f.write(f"\nTraining Params:\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"LR: {lr}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Train Split Ratio: {train_split_ratio}\n")
        f.write(f"Base CLIP Model: {model_path if model_path else model_name}\n")

    print(f"Classification results saved to {result_txt_path}")

    return (
        clip_model,
        linear_probe,
        all_image_features,
        all_image_labels,
        class_names,
        processor,
        val_dataset
    )


# --- GradCAM Functions (Adapted) ---

def vit_reshape_transform(tensor):
    # tensor: [B, 50, 768]
    B, N, C = tensor.shape
    h = w = int((N - 1) ** 0.5)  # 49 => 7x7
    result = tensor[:, 1:, :].reshape(B, h, w, C)  # [B, 7, 7, 768]
    result = result.permute(0, 3, 1, 2).contiguous()  # [B, 768, 7, 7]
    return result


def run_gradcam_vit_linear_probe(clip_model, linear_probe, processor, image_pil):
    image_pil = image_pil.resize((224, 224)) # Ensure correct size
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    input_tensor = inputs["pixel_values"]  # shape [1, 3, H, W]
    img_np = np.array(image_pil).astype(np.float32) / 255.0

    # Wrap model with linear probe
    wrapper = CLIPLinearProbeLogitWrapper(clip_model, linear_probe).to(device)
    wrapper.eval()

    # Get the prediction from the linear probe for this image
    with torch.no_grad():
        logits = wrapper(input_tensor)
        pred_class_idx = logits.argmax(dim=-1).item()
        # print(f"Predicted class index for GradCAM: {pred_class_idx}")

    # Target the layer *within the CLIP vision model*
    # Usually the last attention block or layer norm within it
    try:
        # Common target layers - adjust if needed for specific CLIP variants
        target_layers = [clip_model.vision_model.encoder.layers[-1].layer_norm2]
        # Alternative: target_layers = [clip_model.vision_model.post_layernorm]
    except AttributeError:
        print("Could not find default target layer. Trying final layer norm.")
        try:
            target_layers = [clip_model.vision_model.post_layernorm]
        except AttributeError:
             print("Error: Cannot find suitable target layer for GradCAM in the vision model.")
             return None # Cannot proceed

    # Set up GradCAM
    cam = GradCAM(
        model=wrapper,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform, # Use the reshape function
    )

    # Target the *predicted* class index from the linear probe
    targets = [ClassifierOutputTarget(pred_class_idx)]

    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        if grayscale_cam is None or grayscale_cam.shape[0] == 0:
             print("GradCAM returned empty result.")
             return None
        grayscale_cam = grayscale_cam[0, :] # Take the first (and only) CAM result

        vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return Image.fromarray(vis)
    except Exception as e:
        print(f"Error during GradCAM generation: {e}")
        # This might happen if reshape_transform fails or internal shapes mismatch
        return None


def visualize_gradcam(
    clip_model,
    linear_probe,
    processor,
    val_dataset, # Use validation dataset to pick images
    class_names,
    out_dir="../results/clip_linear_probe/gradcam_pred",
):
    os.makedirs(out_dir, exist_ok=True)

    # Get one image per class from the validation set
    images_to_visualize = {}
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        if item is None: continue
        label_idx = item['label']
        class_name = class_names[label_idx]
        if class_name not in images_to_visualize:
            images_to_visualize[class_name] = item['path']
        if len(images_to_visualize) == len(class_names):
            break # Found one image for each class

    print(f"\nGenerating GradCAM for {len(images_to_visualize)} classes...")
    for class_name, img_path in images_to_visualize.items():
        try:
            image_pil = Image.open(img_path).convert("RGB")
            result_img = run_gradcam_vit_linear_probe(clip_model, linear_probe, processor, image_pil)

            if result_img:
                save_name = f"{class_name}_gradcam_pred.jpg"
                result_img.save(os.path.join(out_dir, save_name))
                print(f"GradCAM (predicted class) for {class_name} => {save_name}")
            else:
                 print(f"Skipped GradCAM for {class_name} due to error.")

        except Exception as e:
            print(f"Error processing GradCAM for {class_name} ({img_path}): {e}")


# --- t-SNE Function (Unchanged logic, just ensure features are passed) ---
def visualize_tsne(
    features, labels, class_names, out_path="../results/clip_linear_probe/tsne_features.svg"
):
    if features is None or features.shape[0] == 0:
        print("No features provided for t-SNE. Skipping.")
        return

    print(f"\nApplying t-SNE on {features.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1), n_iter=1000, init='pca', learning_rate='auto') # Adjust perplexity if needed
    try:
        emb = tsne.fit_transform(features)

        plt.figure(figsize=(12, 10)) # Increased size for legend
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))

        for i, label_idx in enumerate(unique_labels):
             if label_idx >= len(class_names):
                 print(f"Warning: Label index {label_idx} out of bounds for class_names (length {len(class_names)}). Skipping this label in t-SNE plot.")
                 continue
             cls = class_names[label_idx]
             mask = labels == label_idx
             plt.scatter(emb[mask, 0], emb[mask, 1], color=colors[label_idx], label=cls, alpha=0.7, s=15) # Adjusted alpha/size

        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize='small')
        plt.title("t-SNE of CLIP Image Features (Linear Probing Base)")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(out_path, format="svg", bbox_inches='tight')
        plt.close()
        print(f"t-SNE visualization saved at {out_path}")
    except Exception as e:
        print(f"Error during t-SNE: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP models using Linear Probing"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="../data/Aerial_Landscapes", help="Dataset directory"
    )
    parser.add_argument(
        "--base_save_dir", type=str, default="../results/clip_linear_probe", help="Base directory to save results"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train the linear probe"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the linear probe optimizer"
    )
    parser.add_argument(
        "--train_split_ratio", type=float, default=0.8, help="Ratio of data to use for training"
    )
    parser.add_argument(
        "--model_dir", type=str, help="Directory containing fine-tuned CLIP models (expects subdirs with model files)", default=None
    )
    parser.add_argument(
        "--single_model", type=str, help="Path to a specific fine-tuned CLIP model directory", default=None
    )
    args = parser.parse_args()

    if args.single_model:
        model_paths = [args.single_model]
    elif args.model_dir:
        model_paths = [
            os.path.join(args.model_dir, d, "clip_finetuned_final")
            for d in os.listdir(args.model_dir)
            if os.path.isdir(os.path.join(args.model_dir, d, "clip_finetuned_final"))
        ]
    else:
        model_paths = [None]

    for model_path in model_paths:
        if model_path:
            model_name_part = Path(model_path).parent.name
            save_dir = os.path.join(args.base_save_dir, model_name_part)
        else:
            save_dir = os.path.join(args.base_save_dir, "pretrained")
            model_name_part = "pretrained"

        print(f"\n\n===== Running Linear Probing with base model: {model_name_part} =====")
        print(f"Results will be saved in: {save_dir}")

        # --- Run Linear Probing Training and Evaluation ---
        (
            frozen_clip_model,
            trained_linear_probe,
            image_features, # Features from validation set
            image_labels,   # Labels from validation set
            class_names,
            processor,
            val_dataset
        ) = run_clip_linear_probing(
            args.dataset_dir,
            save_dir,
            args.batch_size,
            model_path, # Pass the specific model path
            args.epochs,
            args.lr,
            args.train_split_ratio
        )

        # --- Run Visualizations ---

        # Step 2: Grad-CAM (using predicted class)
        visualize_gradcam(
            frozen_clip_model,
            trained_linear_probe,
            processor,
            val_dataset, # Pass validation dataset
            class_names,
            out_dir=os.path.join(save_dir, "gradcam_pred"),
        )

        # Step 3: T-SNE on validation set features
        visualize_tsne(
            image_features,
            image_labels,
            class_names,
            out_path=os.path.join(save_dir, "tsne_features_val.svg"),
        )

        print(f"All tasks done for model: {model_name_part}")