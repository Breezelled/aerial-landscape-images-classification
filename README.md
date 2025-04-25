<div align="center">
  <h1><b> Aerial Landscape Images Classification </b></h1>
</div>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.6.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## 📌 Overview
This is the SkyView aerial image classification project. We leverage recent advances in pretraining, multimodal learning, and representation learning to build explainable and transferable systems for aerial landscape classification. We also provide comprehensive comparisons with traditional machine learning methods.

Our study presents a unified empirical evaluation of aerial scene classification across traditional, supervised, and self-supervised learning paradigms, using the Skyview dataset as a common benchmark. Results show that supervised deep learning models, particularly ViT and ResNet, consistently outperform traditional handcrafted pipelines in accuracy and feature discriminability.

## 🧠 Key Findings

- **Fine-tuning vs. Linear Probing**: Fine-tuning yields stronger class separation than linear probing, and segmentation-enhanced inputs further improve performance by highlighting salient regions.
- **Self-supervised Approaches**: CLIP demonstrates strong generalization and benefits significantly from lightweight domain adaptation using RSICD dataset.
- **Traditional Methods**: Descriptors such as LBP, when paired with robust classifiers like Random Forests, remain viable in low-data scenarios.
- **Adversarial Training**: Emerges as a highly effective defense strategy, improving model robustness with minimal compromise in clean accuracy.

## 📁 Datasets

### 1. SkyView (Main Task Dataset)
- 15 scene categories (e.g., Forest, River, City, Mountain)
- 800 images per class (12,000 total)
- All images resized to 256×256 pixels
- Curated from AID and NWPU-Resisc45 datasets
- [Kaggle Link](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)

### 2. RSICD (for CLIP Fine-tuning)
- 10,921 image-caption pairs
- Each image annotated with 5 natural language captions
- Enables training vision-language models on remote sensing images
- [RSICD Dataset](https://github.com/201528014227051/RSICD_optimal)

### 3. SSL4EO-S12 (for Domain-specific Pretraining)
- Large-scale multimodal, multitemporal dataset for self-supervised learning
- Built from Sentinel-2 RGB imagery
- High semantic overlap with SkyView classes
- [SSL4EO Website](https://github.com/zhu-xlab/SSL4EO-S12)

## 🧠 Methods

### Traditional Machine Learning
- **Feature Extraction**: SIFT (Scale-Invariant Feature Transform) and LBP (Local Binary Pattern)
- **Classifiers**: SVM, kNN, Random Forest, and Logistic Regression

### Supervised Deep Learning
- **ResNet-50**: Tested with both ImageNet pretraining and remote sensing-specific pretraining (SSL4EO-S12)
- **Vision Transformer (ViT)**: Pretrained on ImageNet-21K
- **SAM2 Segmentation-Enhanced Inputs**: Improved classification by highlighting salient regions and reducing background noise

### Self-supervised Learning
- **CLIP (Contrastive Language-Image Pretraining)**: Zero-shot classification and linear probing
- **Domain-adapted CLIP**: Fine-tuned on RSICD remote sensing captions
- **DINOv2**: Large-scale visual self-supervised learning

### Adversarial Training
- Evaluated model robustness against structured perturbations
- Developed defense strategies through adversarial fine-tuning
- Achieved 98.75% accuracy on adversarial inputs while maintaining 96.46% on clean data

## 🔬 Experiments & Results

| Model                                       | Pretrain      | Setting              | Acc@1 | Acc@3 | Acc@5 | Precision | Recall | F1    |
|---------------------------------------------|---------------|----------------------|-------|-------|-------|-----------|--------|-------|
| **Traditional ML**                           |               |                      |       |       |       |           |        |       |
| SVM (LBP features)                          | None          | —                    | 54.17 | 84.13 | 92.71 | 54.37     | 54.17  | 53.79 |
| kNN (LBP features)                          | None          | —                    | 50.38 | 75.17 | 82.83 | 50.82     | 50.38  | 50.19 |
| Random Forest (LBP features)                | None          | —                    | 57.83 | 84.88 | 93.46 | 57.71     | 57.83  | 57.48 |
| Logistic Regression (LBP features)          | None          | —                    | 32.46 | 63.71 | 78.17 | 29.91     | 32.46  | 27.34 |
| **ResNet-50**                                |               |                      |       |       |       |           |        |       |
| ResNet-50                                   | ImageNet-1K   | Linear probing       | 86.88 | 97.79 | 99.33 | 86.90     | 87.03  | 86.87 |
| ResNet-50                                   | ImageNet-1K   | Fine-tuning          | 95.75 | 99.33 | 99.79 | 95.79     | 95.75  | 95.76 |
| ResNet-50 (with SAM2 masks)                 | ImageNet-1K   | Linear probing       | 89.54 | 98.42 | 99.42 | 89.69     | 89.67  | 89.65 |
| ResNet-50 (with SAM2 masks)                 | ImageNet-1K   | Fine-tuning          | 97.04 | 99.71 | 99.92 | 97.09     | 97.09  | 97.08 |
| ResNet-50                                   | SSL4EO-S12    | Linear probing       | 88.67 | 98.38 | 99.58 | 88.65     | 88.67  | 88.61 |
| ResNet-50                                   | SSL4EO-S12    | Fine-tuning          | 95.54 | 99.21 | 99.67 | 95.64     | 95.54  | 95.54 |
| **ViT**                                      |               |                      |       |       |       |           |        |       |
| ViT-Base                                    | ImageNet-21K  | Linear probing       | 87.46 | 98.00 | 99.50 | 87.52     | 87.49  | 87.46 |
| ViT-Base                                    | ImageNet-21K  | Fine-tuning          | 97.29 | 99.75 | 99.92 | 97.44     | 97.38  | 97.35 |
| **CLIP**                                     |               |                      |       |       |       |           |        |       |
| CLIP-Base (ViT-B/32)                        | WIT-400M      | Zero-shot            | 78.44 | 94.91 | 98.66 | 80.85     | 78.44  | 77.33 |
| CLIP-Base (Fine-tuned w/ RSICD)             | WIT-400M      | Zero-shot            | 90.44 | 98.67 | 99.68 | 91.12     | 90.44  | 90.34 |
| CLIP-Base                                   | WIT-400M      | Linear probing       | 92.54 | 99.21 | 99.88 | 92.69     | 92.68  | 92.58 |
| CLIP-Base (Fine-tuned w/ RSICD)             | WIT-400M      | Linear probing       | 94.29 | 99.67 | 100.0 | 94.39     | 94.38  | 94.30 |
| **DINOv2**                                   |               |                      |       |       |       |           |        |       |
| DINOv2-Base                                 | LVD-142M      | Linear probing       | 88.54 | 97.58 | 99.08 | 89.16     | 88.77  | 88.50 |

## 🧠 Explainability & Visualization

### Feature Understanding
- **t-SNE Visualizations**: Revealed that fine-tuning consistently improves class separability, segmentation enhances structure awareness, and domain-specific pretraining facilitates better generalization.
- **ViT Attention Maps**: Showed model focus areas for both clean and adversarial inputs, explaining classification decisions.
- **Random Forest Feature Importance**: Demonstrated LBP feature effectiveness with a long-tail pattern of feature importance.

### Adversarial Analysis
- **Pixel Modification Impact**: Only 2.89% pixel modification caused 19.33% accuracy drop.
- **Visualization of Different Noise Types**: Compared adversarial, Gaussian, and Salt & Pepper noise effects.
- **Attention Map Shifts**: Revealed how adversarial examples maintain spatial attention but alter feature characteristics.

## 📂 Project Structure
```
project/
├── models/                # ResNet, CLIP, DINOv2, ViT models and ML code
├── utils/                 # Utility functions (mask generator)
├── README.md              # This file
├── requirements.txt       # This file
```

##  Dependencies

Install required libraries:

```bash
pip install -r requirements.txt
```

## ⚙️ How to Run

### CLIP Linear Probing
```bash
>> cd models
# Modify parameters at the bottom of clip_linearprobe.py if needed:
#    --dataset_dir "../data/Aerial_Landscapes"
#    --batch_size 512
#    --epochs 10
#    --lr 1e-3
#    --train_split_ratio 0.8
#    --model_dir /path/to/fine-tuned/model (optional)
>> python clip_linearprobe.py
```
### CLIP Fine-tuning
```bash
>> cd models
# Modify parameters at the bottom of clip_finetune.py if needed:
#    --learning_rate 1e-4
#    --batch_size 512
#    --num_epochs 20
#    --weight_decay 0.001
#    --save_every 1
#    --images_dir ../data/RSICD/RSICD_images
#    --json_path ../data/RSICD/annotations_rsicd/dataset_rsicd.json
>> python clip_finetune.py
```

### CLIP Zero-shot Evaluation
```bash
>> cd models
# Modify parameters at the bottom of clip_zeroshot.py if needed:
#    --dataset_dir "../data/Aerial_Landscapes"
#    --base_save_dir "../results/clip_zeroshot"
#    --batch_size 512
#    --model_dir /path/to/fine-tuned/model (optional)
>> python clip_zeroshot.py
```

### DINOv2 Linear Probing
```bash
>> cd models
# Modify parameters at the bottom of dinov2_linearprobe.py if needed:
#    --dataset_dir "../data/Aerial_Landscapes"
#    --batch_size 512
#    --epochs 10
#    --lr 1e-3
#    --train_split_ratio 0.8
#    --model_path /path/to/pretrained/dinov2 (optional)
>> python dinov2_linearprobe.py
```

### ViT and Adversarial
```bash
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

### Resnet-50 (mask dataset)
```bash
>> cd utils
# step 1: generate mask dataset
# install sam2 from https://github.com/facebookresearch/sam2.git
>> python gen_mask.py Aerial_dataset
# the line above produce a new file Aerial_Landscapes_Masks
# step 2: model on mask dataset
>> cd ..
>> cd models
# For linear-probing
>> python resnet50_linearprobe_on_mask.py --data_dir ../data/Aerial_Landscapes_Masks --epochs 10 --batch_size 256 --lr 0.0001
# For fully fine-tune
>> python resnet50_fine_tune_linearprobe_on_mask.py --data_dir ../data/Aerial_Landscapes_Masks --epochs 10 --batch_size 256 --lr 0.0001
# You can modify the command-line arguments to experiment with different model hyperparameters.
```

### ResNet-50
```bash
# ResNet-50 Linear Probing (frozen encoder, trained classifier)
>> cd models
# Modify parameters in resnet50_linearprobe.py if needed:
#    --dataset_dir "../data/Aerial_Landscapes"
#    --batch_size 256
#    --epochs 10
#    --lr 1e-4
>> python resnet50_linearprobe.py

# ResNet-50 Fine-tuning
>> cd models
# Modify parameters in resnet50_finetune.py if needed:
#    --dataset_dir "../data/Aerial_Landscapes"
#    --batch_size 256
#    --epochs 10
#    --lr 1e-4
>> python resnet50_finetune.py
```

### TorchGeo ResNet-50 (pretrained on SSL4EO-S12)
```bash
# TorchGeo ResNet-50 Linear Probing (SSL4EO-S12 pretraining)
>> cd models
# Modify parameters in torchgeo_linearprobe.py if needed:
#    --dataset_dir "../data/Aerial_Landscapes"
#    --batch_size 256
#    --epochs 10
#    --lr 1e-4
#    --pretrained "ssl4eo-s12"
>> python torchgeo_linearprobe.py

# TorchGeo ResNet-50 Fine-tuning (pretrained on SSL4EO-S12)
>> cd models
# Modify parameters in torchgeo_finetune.py if needed:
#    --dataset_dir "../data/Aerial_Landscapes"
#    --batch_size 256
#    --epochs 10
#    --lr 1e-4
#    --pretrained "ssl4eo-s12"
>> python torchgeo_finetune.py
```

### Run LBP Classification
```bash
cd models/ml
python lbp_classifiers.py
# Extracts 10-D LBP histograms
# Trains all 4 classifiers
# Saves results to `results/lbp_model_results_with_topk.csv` and `lbp_model_detailed_report.txt`
```

### Run SIFT + PCA Classification

```bash
cd models/ml
python sift_pca_classifiers.py
# Extracts 128-D SIFT features
# Applies PCA to reduce to 50-D
# Trains all 4 classifiers
# Saves results to `results/sift_model_results_with_topk.csv` and `sift_model_detailed_report.txt`
```

## 📈 Conclusion and Future Work

Our comprehensive evaluation demonstrates the effectiveness of combining advanced vision models with domain-specific techniques for aerial image classification. Future work could explore:

1. Larger-scale remote sensing datasets for pretraining self-supervised models directly within the target domain
2. Incorporating multimodal data (multispectral, SAR, temporal imagery)
3. Integrating segmentation directly into the training pipeline as a learnable or task-adaptive module

## 🔗 External Resources
- **ResNet**: [Hugging Face](https://huggingface.co/microsoft/resnet-50) | [Paper](https://arxiv.org/abs/1512.03385)
- **ViT**: [Hugging Face](https://huggingface.co/google/vit-base-patch16-224) | [Paper](https://arxiv.org/abs/2010.11929)
- **CLIP**: [Code](https://github.com/openai/CLIP) | [Paper](https://arxiv.org/abs/2103.00020)
- **DINOv2**: [Code](https://github.com/facebookresearch/dinov2) | [Paper](https://arxiv.org/abs/2304.07193)
- **RSICD Dataset**: [Code](https://github.com/201528014227051/RSICD_optimal) | [Paper](https://arxiv.org/abs/1712.07835)
- **SAM2**: [Code](https://github.com/facebookresearch/sam2.git) | [Paper](https://arxiv.org/abs/2408.00714)
- **TorchGeo**: [Code](https://github.com/torchgeo/torchgeo) | [Paper](https://dl.acm.org/doi/10.1145/3557915.3560953)