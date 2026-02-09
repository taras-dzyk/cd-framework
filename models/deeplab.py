import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DeepLabV3PlusCD(nn.Module):
    def __init__(self, encoder_name="resnet34"):
        super(DeepLabV3PlusCD, self).__init__()
    
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=6,
            classes=1,
            activation='sigmoid'
        )

    def forward(self, x1, x2):

        x = torch.cat([x1, x2], dim=1)
        return self.model(x)