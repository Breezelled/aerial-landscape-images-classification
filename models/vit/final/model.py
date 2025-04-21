import torch
import torch.nn as nn
from transformers import ViTModel,ViTFeatureExtractor


class ViTForImageClassification(nn.Module):
    def __init__(self, num_classes):
        super(ViTForImageClassification, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.image_size = 224
        
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            output_attentions=True
        )

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_token)
        return logits,outputs.attentions