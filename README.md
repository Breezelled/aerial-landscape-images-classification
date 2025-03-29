# Aerial Landscape Images Classification

## Advanced Method Development

### 📌 Overview
This is the **Advanced Method Development** section of our SkyView aerial image classification project. The goal is to leverage recent advances in pretraining, multimodal learning, and representation learning to build explainable and transferable systems.

This section includes:

- **Zero-shot classification using CLIP**, which aligns images and natural language prompts.
- **CLIP fine-tuning using RSICD**, a captioned remote sensing dataset, to enhance domain alignment.
- **Linear probing with DINOv2 and BigEarthNet-pretrained ViT**, to assess feature quality for remote sensing imagery.

All experiments are built around the SkyView dataset, and results from this section contribute to the overall understanding of advanced methods in remote sensing scene classification.

---

### 📁 Datasets

#### 1. SkyView (Main Task Dataset)
- 15 scene categories (e.g. Forest, River, City, Mountain)
- 800 images per class
- [Kaggle Link](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)

#### 2. RSICD (for CLIP Fine-tuning)
- Each image annotated with 5 natural language captions
- Enables training vision-language models on remote sensing images
- [RSICD Website]([https://captain-whu.github.io/RSICD-dataset](https://github.com/201528014227051/RSICD_optimal))

#### 3. BigEarthNet v2.0 (for DINOv2 Finetuneing)
- Over 500k Sentinel-2 image patches
- High semantic overlap with SkyView classes
- [BigEarthNet](https://bigearth.net)

---

### 🧠 Methods

#### 🔹 CLIP Zero-shot Classification
- Use OpenAI's CLIP to match SkyView images with class prompts
- Prompts like: `"A satellite image of a {label}"`
- Outputs top-k predicted labels based on image-text similarity
- ✅ *Interpretable and no training required*

#### 🔹 CLIP Fine-tuned on RSICD
- Fine-tune CLIP image encoder using image-caption pairs from RSICD
- Each image paired with 5 caption variations
- Improve alignment for aerial scenes with remote sensing domain captions
- ✅ *Improves CLIP's performance on satellite image semantics*

#### 🔹 DINOv2 + Linear Probing
- Use `vit_base_patch8_224-s2-v0.1.1` pretrained on BigEarthNet
- Freeze ViT weights and extract image embeddings for SkyView
- Train Logistic Regression or SVM for classification
- ✅ *Evaluates quality of semantic visual features learned from remote sensing*

---

### 🔬 Experiments & Results

| Model                      | Fine-tuned? | Method            | Accuracy (SkyView) |
|----------------------------|-------------|-------------------|---------------------|
| CLIP (ViT-B/32)            | ❌ No       | Zero-shot         | TBD                 |
| CLIP (Fine-tuned w/ RSICD) | ✅ Yes      | Zero-shot         | TBD                 |
| ViT (BigEarthNet-S2)       | ❌ No       | Linear probing     | TBD                 |
| DINOv2 (Fine-tuned w/ BigEarthNet-S2)  | ✅ Yes       | Linear probing     | TBD                 |

> ⚠️ TODO: Replace TBD with final experimental results

---

### 🧠 Explainability & Visualization
- **CLIP**: Top-k textual predictions show model reasoning
- **CLIP + GradCAM**: Visualize which region contributed to decision
- **DINOv2**: Feature embedding visualized using t-SNE / PCA to show class clustering

---

### 📈 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Per-class performance analysis
- Confusion matrix and feature distribution plots

---

### 📂 Project Structure
```
project/
├── data/                  # SkyView, RSICD, BigEarthNet subsets
├── models/                # CLIP, DINOv2, ViT models
├── scripts/               # Feature extraction, training, evaluation
├── results/               # Visualizations, confusion matrix, metrics
├── README.md              # This file (Advanced Methods)
```

> ⚠️ TODO: Replace structure example
---

### ⚙️ How to Run

```bash
```

> ⚠️ TODO: Add running code

---

### 🔗 External Resources
- **CLIP**: [Code](https://github.com/openai/CLIP) | [Paper](https://arxiv.org/abs/2103.00020)
- **DINOv2**: [Code](https://github.com/facebookresearch/dinov2) | [Paper](https://arxiv.org/abs/2304.07193)
- **BigEarthNet v2.0**: [Website](https://bigearth.net) | [Paper](https://arxiv.org/abs/2407.03653)
- **RSICD Dataset**: [Code](https://github.com/201528014227051/RSICD_optimal) | [Paper](https://arxiv.org/abs/1712.07835)

---

