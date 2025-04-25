# Aerial Landscape Images Classification

### 📌 Overview
This is the SkyView aerial image classification project. The goal is to leverage recent advances in pretraining, multimodal learning, and representation learning to build explainable and transferable systems. And also compare with traditional methods like ResNet and ML methods.

Including:

- **Traditional machine learning models**
- **Resnet**
- **ViT**
- **Zero-shot classification using CLIP**, which aligns images and natural language prompts.
- **CLIP fine-tuning using RSICD**, a captioned remote sensing dataset, to enhance domain alignment.
- **Linear probing with DINOv2 and TorchGEO-pretrained ViT**, to assess feature quality for remote sensing imagery.

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


| Model                                       | Fine-tuned?  | Method         | Acc@1 | Acc@3 | Acc@5 | Precision | Recall | F1    |
|---------------------------------------------|--------------|----------------|-------|-------|-------|-----------|--------|-------|
| CLIP (ViT-B/32)                             | ❌ No        | Zero-shot      | 78.44 | 94.91 | 98.66 | 80.85     | 78.44  | 77.33 |
| CLIP (Fine-tuned w/ RSICD)                  | ✅ Yes       | Zero-shot      | 90.56 | 98.66 | 99.66 | 91.22     | 90.56  | 90.45 |
| ViT (vit-base-patch16-224-in21k)            | ❌ No        | Linear probing | 87.46 | 98.00 | 99.50 | 87.52     | 87.49  | 87.46 |
| ViT (Fine-tuned vit-base-patch16-224-in21k) | ✅ Yes       | Linear probing | 97.29 | 99.75 | 99.92 | 97.44     | 97.38  | 97.35 |
| DINOv2 (Fine-tuned w/ TorchGEO)             | ✅ Yes       | Linear probing | TBD   |       |       |           |        |       |
| Resnet50 (linearprobe on mask)              | ❌ No        | Linear probing | 89.54 | 98.42 | 99.42 | 89.69     | 89.67  | 89.65 |
| Resnet50 (finetune on mask)                 | ✅ Yes       | Linear probing | 97.04 | 99.71 | 99.92 | 97.09     | 97.09  | 97.08 |

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
# ViT and Adversarial
>> cd vit
# Change the parameter at the bottom of train.py
#    train_model(model_type='ViT',
#                image_dir="./Aerial_Landscapes_Adv", # training dataset path
#                save_path="./save", # save pt files
#                train_type='frozen', # frozen or all
#                weight_path=None, # load pre-trained weight
#                batch_size=16,
#                epochs=60,
#                save_per_epoch=2)
>> python train.py #Train ViT Model
# Change the parameter at the bottom of evaluate.py
#    get_attention( #
#        model_type='ViT',
#        saved_path1=saved_path1, # load model1 from saved_path1
#        saved_path2=saved_path2, # load model2 from saved_path2
#        model1_name=model_name1,
#        model2_name=model_name2,
#        epoch1=epoch1, # choose the epoch1.pt for model1
#        epoch2=epoch2, # choose the epoch2.pt for model2
#        image_dir1="./Aerial_Landscapes", # choose the test dataset1 for model1
#        image_dir2="./Aerial_Landscapes", # choose the test dataset2 for model2
#        image_name1='Original',
#        image_name2='Original',
#        ex_name='final',
#        batch_size=16,
#        heatmap=True) # whether to generate the heatmap
>> python evaluate.py # Evaluate ViT Model
# Just run the ad_train.py and ad_eval.py, all the parameters have been set. 
>> python ad_train.py # Train Adversarial Model
>> python ad_eval.py # Generate Adversarial Image Instances
```

```bash
# Resnet-50 (mask dataset)
>> cd utils
# step 1: generate mask dataset
# install sam2 from https://github.com/facebookresearch/sam2.git
>> python gen_mask.py Aerial_dataset
# the line above produce a new file Aerial_Landscapes_Masks
# step 2: model on mask dataset
>> cd ..
>> cd modles
# For linear-probing
>> python resnet50_linearprobe_on_mask.py --data_dir ../data/Aerial_Landscapes_Masks --epochs 10 --batch_size 256 --lr 0.0001
# For fully fine-tune
>> python resnet50_fine_tune_linearprobe_on_mask.py --data_dir ../data/Aerial_Landscapes_Masks --epochs 10 --batch_size 256 --lr 0.0001
# You can modify the command-line arguments to experiment with different model hyperparameters.
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

