import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split


class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.root = root

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        filename = os.path.splitext(os.path.basename(path))[0]  # 获取文件名并去除扩展名
        return img, label, filename

def get_data_loader(image_dir,
                    image_size,
                    expected_mean,
                    expected_std,
                    batch_size,
                    per_test=0.2,
                    per_val=0.0,
                    random_seed=42,
                    only_test=False):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=expected_mean, std=expected_std),
    ])

    full_dataset = ImageFolderWithPaths(root=image_dir, transform=transform)
    val_size = int(per_val * len(full_dataset))
    test_size = int(per_test * len(full_dataset))
    train_size = len(full_dataset) - val_size - test_size

    if only_test:
        test_loader = DataLoader(full_dataset,batch_size=batch_size, shuffle=False)
        return 0, 0, test_loader
    else:
        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, train_loader, test_loader

if __name__ == "__main__":
    from transformers import ViTModel, ViTFeatureExtractor
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    train_loader, val_loader, test_loader = get_data_loader(image_dir="./Aerial_Landscapes",
                                                            image_size=42,
                                                            expected_mean=feature_extractor.image_mean,
                                                            expected_std=feature_extractor.image_std,
                                                            batch_size=16)