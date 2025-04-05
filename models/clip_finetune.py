import os
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import (
    CLIPProcessor, 
    CLIPModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import argparse

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on RSICD dataset")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer")
    parser.add_argument("--save_every", type=int, default=1, help="Save model every n epochs")
    parser.add_argument("--images_dir", type=str, default="../data/RSICD/RSICD_images", help="Directory with RSICD images")
    parser.add_argument("--json_path", type=str, default="../data/RSICD/annotations_rsicd/dataset_rsicd.json", help="Path to RSICD annotations")
    parser.add_argument("--output_dir", type=str, default="../results/clip_finetuned_trainer", help="Base output directory")
    return parser.parse_args()

# Custom dataset class
class RSICDDataset(Dataset):
    def __init__(self, images_dir, json_path, processor, split="train"):
        self.processor = processor
        self.images_dir = images_dir
        self.split = split
        
        # Load annotations
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        for item in data['images']:
            if item['split'] == split:
                image_filename = item['filename']
                image_path = os.path.join(images_dir, image_filename)
                
                # Get the first caption for each image
                if len(item['sentences']) > 0:
                    caption = item['sentences'][0]['raw']
                    
                    if os.path.exists(image_path):
                        self.samples.append({
                            'image_path': image_path,
                            'caption': caption
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        caption = sample['caption']
        
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        inputs["return_loss"] = True
        
        return inputs

def collate_fn(examples):
    keys = examples[0].keys()
    
    batch = {k: torch.stack([example[k] for example in examples]) 
             for k in keys if k != "return_loss"}
    
    batch["return_loss"] = True
    
    return batch


def fine_tune_clip_with_trainer(
    images_dir="../data/RSICD/RSICD_images",
    json_path="../data/RSICD/annotations_rsicd/dataset_rsicd.json",
    output_dir="../results/clip_finetuned_trainer",
    batch_size=128,
    num_epochs=10,
    learning_rate=1e-7,
    weight_decay=0.001,
    save_every=1,
    eval_steps=100,
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    def set_model_to_eval_mode(model=model):
        model.eval()
        return model
    
    # Create datasets
    train_dataset = RSICDDataset(
        images_dir=images_dir,
        json_path=json_path,
        processor=processor,
        split="train"
    )
    
    val_dataset = RSICDDataset(
        images_dir=images_dir,
        json_path=json_path,
        processor=processor,
        split="val"
    )
    
    print(f"Traning samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        # weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        eval_steps=save_every * (len(train_dataset) // batch_size),
        save_strategy="steps",
        save_steps=save_every * (len(train_dataset) // batch_size),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fp16=True,
    )
    
    # Set model to evaluation mode before training
    model = set_model_to_eval_mode(model)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        # We don't use compute_metrics because CLIP evaluation needs to be calculated at batch level
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Add hook to reset model to eval mode at the beginning of each epoch
    original_train = trainer.train
    
    def train_with_eval_mode(*args, **kwargs):
        set_model_to_eval_mode()
        return original_train(*args, **kwargs)
    
    trainer.train = train_with_eval_mode
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(output_dir, "clip_finetuned_final"))
    
    # Plot training history
    if trainer.state.log_history:
        train_losses = [x.get("loss") for x in trainer.state.log_history if x.get("loss") is not None]
        eval_losses = [x.get("eval_loss") for x in trainer.state.log_history if x.get("eval_loss") is not None]
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        
        if train_losses:
            plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
        
        if eval_losses:
            # Adjust x-axis position based on evaluation steps
            steps_per_eval = eval_steps
            eval_x = [i * steps_per_eval for i in range(len(eval_losses))]
            plt.plot(eval_x, eval_losses, label='Eval Loss')
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    return model, processor

# Evaluate model on test set
def evaluate_clip_on_test(
    model,
    processor,
    images_dir="../data/RSICD/RSICD_images",
    json_path="../data/RSICD/annotations_rsicd/dataset_rsicd.json",
    output_dir="../results/clip_finetuned_trainer",
    batch_size=128
):
    test_dataset = RSICDDataset(
        images_dir=images_dir,
        json_path=json_path,
        processor=processor,
        split="test"
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Calculate Recall@K metrics
    recall_at_1 = 0
    recall_at_5 = 0
    total_samples = len(test_dataset)
    
    # Encode all images and texts in test set
    all_image_features = []
    all_text_features = []
    all_captions = []
    
    model.to(device)
    model.eval()
    
    # Get features for all test images
    for idx in tqdm(range(len(test_dataset)), desc="encoding test samples"):
        sample = test_dataset[idx]
        image_path = test_dataset.samples[idx]['image_path']
        caption = test_dataset.samples[idx]['caption']
        all_captions.append(caption)
        
        with torch.no_grad():
            # Encode image
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features.cpu().numpy())
            
            # Encode text
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features.cpu().numpy())
    
    all_image_features = np.vstack(all_image_features)
    all_text_features = np.vstack(all_text_features)
    
    # Calculate image-to-text similarity matrix
    similarity_scores = np.matmul(all_image_features, all_text_features.T)
    
    # Calculate image-to-text Recall@K
    for i in range(total_samples):
        # Get similarity scores for current image
        scores = similarity_scores[i]
        
        # Find indices of most similar texts
        top_indices = np.argsort(scores)[::-1]
        
        # Calculate Recall@1 and Recall@5
        recall_at_1 += 1 if i in top_indices[:1] else 0
        recall_at_5 += 1 if i in top_indices[:5] else 0
    
    # Calculate average recall
    recall_at_1 /= total_samples
    recall_at_5 /= total_samples
    
    print(f"Test results:")
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@5: {recall_at_5:.4f}")
    
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Recall@1: {recall_at_1:.4f}\n")
        f.write(f"Recall@5: {recall_at_5:.4f}\n")
    
    return recall_at_1, recall_at_5

def visualize_retrieval_examples(
    model,
    processor,
    images_dir="../data/RSICD/RSICD_images",
    json_path="../data/RSICD/annotations_rsicd/dataset_rsicd.json",
    num_examples=5,
    output_dir="../results/clip_finetuned_trainer"
):
    test_dataset = RSICDDataset(
        images_dir=images_dir,
        json_path=json_path,
        processor=processor,
        split="test"
    )
    
    example_indices = np.random.choice(len(test_dataset), min(num_examples, len(test_dataset)), replace=False)
    
    # Get all test captions
    all_captions = [sample['caption'] for sample in test_dataset.samples]
    
    # Create directory for retrieval examples
    retrieval_dir = os.path.join(output_dir, 'retrieval_examples')
    os.makedirs(retrieval_dir, exist_ok=True)
    
    model.to(device)
    model.eval()
    
    # Encode all texts
    text_inputs = processor(
        text=all_captions,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    for i, idx in enumerate(example_indices):
        sample = test_dataset[idx]
        image_path = test_dataset.samples[idx]['image_path']
        query_caption = test_dataset.samples[idx]['caption']
        
        # Get image features
        with torch.no_grad():
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
            image_features = model.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            similarity = (image_features @ text_features.T).squeeze(0)
            
            # Get top 5 results
            top_indices = similarity.topk(5)[1].cpu().numpy()
            top_captions = [all_captions[idx] for idx in top_indices]
            top_scores = similarity[top_indices].cpu().numpy()
        
        # Load original image
        image = Image.open(image_path).convert("RGB")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Query Image")
        
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.text(0, 0.5, f"original annotation:\n{query_caption}\n\n Top 5 annotation:", 
                fontsize=12, fontweight='bold')
        
        y_pos = 0.7
        for j, (caption, score) in enumerate(zip(top_captions, top_scores)):
            color = 'green' if caption == query_caption else 'black'
            plt.text(0, y_pos, f"{j+1}. {caption}", fontsize=10, color=color)
            plt.text(0.8, y_pos, f"{score:.4f}", fontsize=10, color=color)
            y_pos += 0.1
        
        plt.tight_layout()
        plt.savefig(os.path.join(retrieval_dir, f"example_{i+1}.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    args = parse_args()
    
    param_dir = f"lr{args.learning_rate}_bs{args.batch_size}_ep{args.num_epochs}"
    output_dir = os.path.join(args.output_dir, param_dir)
    
    model, processor = fine_tune_clip_with_trainer(
        images_dir=args.images_dir,
        json_path=args.json_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_every=1,
        eval_steps=args.save_every * (len(RSICDDataset(args.images_dir, args.json_path, CLIPProcessor.from_pretrained(model_name), "train")) // args.batch_size),
    )
    
    evaluate_clip_on_test(
        model,
        processor,
        images_dir=args.images_dir,
        json_path=args.json_path,
        output_dir=output_dir
    )
    
    visualize_retrieval_examples(
        model,
        processor,
        images_dir=args.images_dir,
        json_path=args.json_path,
        num_examples=10,
        output_dir=output_dir
    )
    
    print(args)
    print("Finished CLIP finetune.")