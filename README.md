# Aerial Landscape Images Classification

### 📌 Overview
This is the SkyView aerial image classification project. The goal is to leverage recent advances in pretraining, multimodal learning, and representation learning to build explainable and transferable systems. And also compare with traditional methods like ResNet and ML methods.

Including:

- **Traditional machine learning models**
- **Resnet**
- **ViT**
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
- [RSICD Dataset](https://github.com/201528014227051/RSICD_optimal)

#### 3. BigEarthNet v2.0 (for DINOv2 Finetuneing)
- Over 500k Sentinel-2 image patches
- High semantic overlap with SkyView classes
- [BigEarthNet Website](https://bigearth.net)

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

#### 🔹 Resnet50 + image mask + Linear Probing
- Use resnet50 pretrained on ImageNet
- Extract features from masked remote sensing images (e.g., using SAM2-generated masks)
- Freeze ResNet-50 weights and use extracted features for classification
- ✅ *Evaluates the effectiveness of masked image features in remote sensing classification*


#### 🔹 Resnet50 + image mask + Fine tuned
- Use resnet50 pretrained on ImageNet
- Extract features from masked remote sensing images (e.g., using SAM2-generated masks)
- Fine-tune all weights jointly with the classifier head
- ✅ *Evaluates the effectiveness of masked image features in remote sensing classification*
---

### 🔬 Experiments & Results


| Model                                 | Fine-tuned? | Method         | Acc@1 | Acc@3 | Acc@5 | Precision | Recall | F1    |
| ------------------------------------- | ----------- | -------------- | ----- | ----- | ----- | --------- | ------ | ----- |
| CLIP (ViT-B/32)                       | ❌ No        | Zero-shot      | 78.44 | 94.91 | 98.66 | 80.85     | 78.44  | 77.33 |
| CLIP (Fine-tuned w/ RSICD)            | ✅ Yes       | Zero-shot      | 90.56 | 98.66 | 99.66 | 91.22     | 90.56  | 90.45 |
| ViT (BigEarthNet-S2)                  | ❌ No        | Linear probing | TBD   |       |       |           |        |       |
| DINOv2 (Fine-tuned w/ BigEarthNet-S2) | ✅ Yes       | Linear probing | TBD   |       |       |           |        |       |
| Resnet50 (linearprobe on mask)        | ❌ No        | Linear probing | 89.54 | 98.42 | 99.42 | 89.69     | 89.67  | 89.65 |
| Resnet50 (finetune on mask)           | ✅ Yes       | Linear probing | 97.04 | 99.71 | 99.92 | 97.09     | 97.09  | 97.08 |

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
├── models/                # CLIP, DINOv2, ViT models...
├── scripts/               # Shell scripts for feature extraction, training, evaluation
├── utils/                 # Utility functions (mask generator...)
├── results/               # Visualizations, confusion matrix, metrics...
├── README.md              # This file
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
- **SAM2**: [Code](https://github.com/facebookresearch/sam2.git) | [Paper](https://arxiv.org/abs/2408.00714)

---

