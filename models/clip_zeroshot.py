import os
import numpy as np
import torch
import torch.nn as nn
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
from torch.utils.data import Dataset, DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import argparse
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"


class AerialImageDataset(Dataset):
    def __init__(self, dataset_dir, processor):
        self.processor = processor
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(dataset_dir))

        for cls_idx, class_name in enumerate(self.class_names):
            folder = os.path.join(dataset_dir, class_name)
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
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        # Remove batch dimension from HF processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return {
            "inputs": inputs,
            "label": self.labels[idx],
            "path": image_path,
        }


class CLIPTop1LogitWrapper(nn.Module):

    def __init__(self, clip_model, text_embeds):
        super().__init__()
        self.clip_model = clip_model
        # Register text_embeds as a buffer (no gradient) to keep them on the same device
        self.register_buffer("text_embeds", text_embeds)

    def forward(self, pixel_values):
        """
        pixel_values: shape (B, 3, H, W)
        We'll replicate the steps to get image_features and do dot-product w/ text_embeds.
        Return shape (B,): each sample's max logit among all text prompts.
        """
        vision_out = self.clip_model.vision_model(pixel_values=pixel_values)
        last_hidden = vision_out[0]  # shape (B, seq_len, hidden_dim)

        pooled = self.clip_model.visual_projection(
            self.clip_model.vision_model.post_layernorm(last_hidden[:, 0, :])
        )
        image_embeds = pooled / pooled.norm(p=2, dim=-1, keepdim=True)  # shape (B, dim)

        # Dot product
        sim = image_embeds @ self.text_embeds.T  # shape (B, n_prompts)
        return sim


def collate_fn(batch):
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


def run_clip_zero_shot(
    dataset_dir="../data/Aerial_Landscapes",
    save_dir="../results/clip_zeroshot",
    batch_size=32,
    model_path=None,
):
    os.makedirs(save_dir, exist_ok=True)

    if model_path and os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(f"Loading fine-tuned model from {model_path}")
        model = CLIPModel.from_pretrained(model_path).to(device)
    else:
        print(f"Loading pre-trained model {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(device)

    processor = CLIPProcessor.from_pretrained(model_name)

    class_names = sorted(os.listdir(dataset_dir))
    prompts = [f"A satellite image of a {c.lower()}" for c in class_names]

    # Encode text prompts
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    dataset = AerialImageDataset(dataset_dir, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    true_labels = []
    pred_labels = []
    top1_count = top3_count = top5_count = 0
    total = 0

    per_class_correct = {cls: 0 for cls in class_names}
    per_class_total = {cls: 0 for cls in class_names}

    # Store features for t-SNE
    all_image_features = []
    all_image_labels = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        batch_labels = batch["labels"].numpy()

        with torch.no_grad():
            image_features = model.get_image_features(**batch_inputs)
            batch_features_np = image_features.cpu().numpy()
            for i, feat in enumerate(batch_features_np):
                all_image_features.append(feat)
                all_image_labels.append(batch_labels[i])

            # normalize
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )
            similarity = image_features @ text_features.T
            probs = similarity.softmax(dim=-1)

        probs_np = probs.cpu().numpy()

        for i, (prob, label) in enumerate(zip(probs_np, batch_labels)):
            class_name = class_names[label]

            topk = np.argsort(prob)[::-1][:5].tolist()

            total += 1
            true_labels.append(label)
            pred_labels.append(topk[0])

            if topk[0] == label:
                top1_count += 1
                per_class_correct[class_name] += 1
            if label in topk[:3]:
                top3_count += 1
            if label in topk:
                top5_count += 1
            per_class_total[class_name] += 1

    print("Classification Report (Macro):")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    acc1 = top1_count / total
    acc3 = top3_count / total
    acc5 = top5_count / total
    print(f"\nTop-1 Accuracy: {acc1}")
    print(f"Top-3 Accuracy: {acc3}")
    print(f"Top-5 Accuracy: {acc5}")

    macro_precision = precision_score(true_labels, pred_labels, average="macro")
    macro_recall = recall_score(true_labels, pred_labels, average="macro")
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    print(f"Macro Precision: {macro_precision}")
    print(f"Macro Recall:    {macro_recall}")
    print(f"Macro F1-score:  {macro_f1}")

    result_txt_path = os.path.join(save_dir, "classification_results.txt")
    with open(result_txt_path, "w") as f:
        f.write("Classification Report (Macro):\n")
        f.write(
            classification_report(true_labels, pred_labels, target_names=class_names)
        )
        f.write(f"\nTop-1 Accuracy: {acc1}\n")
        f.write(f"Top-3 Accuracy: {acc3}\n")
        f.write(f"Top-5 Accuracy: {acc5}\n")
        f.write(f"Macro Precision: {macro_precision}\n")
        f.write(f"Macro Recall:    {macro_recall}\n")
        f.write(f"Macro F1-score:  {macro_f1}\n")

    print(f"Classification results saved to {result_txt_path}")

    # per-class rep image
    rep_images = []
    for cls in class_names:
        fold = os.path.join(dataset_dir, cls)
        found_img = None
        for f in os.listdir(fold):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                found_img = os.path.join(fold, f)
                break
        rep_images.append((cls, found_img))

        fig, axs = plt.subplots(3, 5, figsize=(25, 15))
        axs = axs.flatten()

        for idx, (cls, img_path) in enumerate(rep_images):
            image_pil = Image.open(img_path).convert("RGB")
            inp = processor(images=image_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model.get_image_features(**inp)
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
                sim = feat @ text_features.T
                p = sim.softmax(dim=-1).squeeze().cpu().numpy()

            topk_ids = p.argsort()[::-1][:5]
            topk_lbl = [class_names[j] for j in topk_ids]
            topk_val = p[topk_ids]

            ax = axs[idx]
            ax.axis("off")
            ax.imshow(image_pil)
            ax.set_title(cls, fontsize=12)

            ax_bar = ax.inset_axes([0.0, -0.35, 1.0, 0.25])
            bar_colors = [
                "limegreen" if lbl == cls else "skyblue" for lbl in topk_lbl[::-1]
            ]
            y_pos = np.arange(len(topk_lbl))
            ax_bar.barh(y_pos, topk_val[::-1], color=bar_colors)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(topk_lbl[::-1], fontsize=8)
            ax_bar.set_xlim(0, 0.1)

            for i, v in enumerate(topk_val[::-1]):
                ax_bar.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=7)

        plt.subplots_adjust(hspace=1.0)
        plt.suptitle("Top-5 Predictions per Class (CLIP Zero-shot)", fontsize=16)
        plt.savefig(
            os.path.join(save_dir, "top5_per_class_visualization.svg"), format="svg"
        )
        plt.close()

    return (
        text_features.cpu(),
        np.array(all_image_features),
        np.array(all_image_labels),
        class_names,
        model,
    )


def vit_reshape_transform(tensor):
    # tensor: [B, 50, 768]
    B, N, C = tensor.shape
    h = w = int((N - 1) ** 0.5)  # 49 => 7x7
    result = tensor[:, 1:, :].reshape(B, h, w, C)  # [B, 7, 7, 768]
    result = result.permute(0, 3, 1, 2).contiguous()  # [B, 768, 7, 7]
    return result


def run_gradcam_vit_top1(clip_model, text_features, image_pil):
    processor = CLIPProcessor.from_pretrained(model_name)
    image_pil = image_pil.resize((224, 224))
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    input_tensor = inputs["pixel_values"]  # shape [1, 3, H, W]
    img_np = np.array(image_pil).astype(np.float32) / 255.0

    # wrap model
    wrapper = CLIPTop1LogitWrapper(clip_model, text_features).to(device)
    wrapper.eval()

    target_layers = [clip_model.vision_model.encoder.layers[-1].layer_norm2]

    # set up GradCAM
    cam = GradCAM(
        model=wrapper,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform,
    )

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[
        0
    ]
    vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return Image.fromarray(vis)


def visualize_gradcam(
    clip_model,
    text_features,
    dataset_dir,
    out_dir="../results/clip_zeroshot/gradcam_top1",
):
    os.makedirs(out_dir, exist_ok=True)
    class_names = sorted(os.listdir(dataset_dir))

    for class_name in class_names:
        folder = os.path.join(dataset_dir, class_name)
        gradcam_img_path = None

        # pick the first image
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                gradcam_img_path = os.path.join(folder, f)
                break
        if gradcam_img_path is None:
            continue

        image_pil = Image.open(gradcam_img_path).convert("RGB")
        result = run_gradcam_vit_top1(clip_model, text_features, image_pil)
        save_name = f"{class_name}_gradcam_top1.jpg"
        result.save(os.path.join(out_dir, save_name))
        print(f"GradCAM (top1) for {class_name} => {save_name}")


def visualize_tsne(
    features, labels, class_names, out_path="../results/clip_zeroshot/tsne_features.svg"
):
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    for i, cls in enumerate(class_names):
        mask = labels == i
        plt.scatter(emb[mask, 0], emb[mask, 1], c=[colors[i]], label=cls, alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE of CLIP zeroshot Image Features")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"t-SNE visualization saved at {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP models on zero-shot classification"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../data/Aerial_Landscapes",
        help="Dataset directory",
    )
    parser.add_argument(
        "--base_save_dir",
        type=str,
        default="../results/clip_zeroshot",
        help="Base directory to save results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for inference"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing fine-tuned models",
        default=None,
    )
    parser.add_argument(
        "--single_model",
        type=str,
        help="Path to a single fine-tuned model",
        default=None,
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

        print(f"\n\n===== Evaluating model: {model_name_part} =====")

        # Step1: Zero-shot classification
        (
            text_features,
            image_features,
            labels,
            class_names,
            clip_model,
        ) = run_clip_zero_shot(args.dataset_dir, save_dir, args.batch_size, model_path)

        # Step2: Grad-CAM top-1 for each class
        visualize_gradcam(
            clip_model,
            text_features,
            args.dataset_dir,
            out_dir=os.path.join(save_dir, "gradcam_top1"),
        )

        # Step3: T-SNE
        visualize_tsne(
            image_features,
            labels,
            class_names,
            out_path=os.path.join(save_dir, "tsne_features.svg"),
        )

        print(f"All tasks done for model: {model_name_part}")
