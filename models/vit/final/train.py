import os
import torch
import pandas as pd
import torch.nn as nn
from model import ViTForImageClassification
from dataset import get_data_loader

def train_model(model_type='ViT',
                image_dir="./Aerial_Landscapes",
                save_path="./save",
                train_type='all',
                weight_path=None,
                batch_size=16,
                epochs=20,
                save_per_epoch=5):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Settings
    num_classes = len(os.listdir(image_dir))
    if model_type == 'ViT':
        model = ViTForImageClassification(num_classes=num_classes).to(device)
        image_size = model.image_size
        image_mean = model.feature_extractor.image_mean
        image_std = model.feature_extractor.image_std
    else:
        raise ValueError('model_type')
        
    if 'all' in train_type.lower():
        pass
    elif 'frozen' in train_type.lower():
        for name, param in model.vit.named_parameters():
            param.requires_grad = False
    else:
        raise ValueError('train_type')

    if weight_path:
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load Dataset
    train_loader, val_loader, test_loader = get_data_loader(image_dir=image_dir,
                                                            image_size=image_size,
                                                            expected_mean=image_mean,
                                                            expected_std=image_std,
                                                            batch_size=batch_size)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Mkdir
    save_dir = os.path.dirname(f"{save_path}/{model_type}_{train_type}/")
    os.makedirs(save_dir, exist_ok=True)

    # Train
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for i, (images, labels,_) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            predicts,_ = model(images)
            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i+1) % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i+1}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")

        train_losses.append(total_loss/len(train_loader))
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        for i, (images, labels,_) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            predicts,_ = model(images)
            loss = criterion(predicts, labels)

            val_loss += loss.item()
        val_losses.append(val_loss/len(test_loader))

        if (epoch+1) % save_per_epoch == 0:
            # Save Checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader)
            }
            torch.save(checkpoint, f"{save_path}/{model_type}_{train_type}/{epoch+1}.pth" )
            print(f"Checkpoint saved at {save_path}/{model_type}_{train_type}/{epoch+1}.pth")

    loss_df = pd.DataFrame({
        'train_loss': train_losses,
        'val_loss': val_losses
    })

    loss_df.to_csv(f"{save_path}/{model_type}_{train_type}/losses.csv", index=False)

# adv is 15
if __name__ == "__main__":
    train_model(model_type='ViT',
                image_dir="./Aerial_Landscapes_Adv",
                save_path="./save",
                train_type='frozen_OA',
                weight_path='./save/ViT_frozen/60.pth',
                batch_size=16,
                epochs=60,
                save_per_epoch=5)
    # train_model(model_type='ViT',
    #             image_dir="./Aerial_Landscapes_Adv",
    #             save_path="./save",
    #             train_type='all_OA',
    #             weight_path='./save/ViT_all/20.pth',
    #             batch_size=16,
    #             epochs=5,
    #             save_per_epoch=1)
    train_model(model_type='ViT',
                image_dir="./Aerial_Landscapes_Adv",
                save_path="./save",
                train_type='frozen_A',
                weight_path=None,
                batch_size=16,
                epochs=60,
                save_per_epoch=5)
    # train_model(model_type='ViT',
    #             image_dir="./Aerial_Landscapes_Adv",
    #             save_path="./save",
    #             train_type='all_A',
    #             weight_path=None,
    #             batch_size=16,
    #             epochs=10,
    #             save_per_epoch=1)