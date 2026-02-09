import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class ResNetUNet(nn.Module):
    def __init__(self, encoder_name="resnet18", pretrained=True):
        super(ResNetUNet, self).__init__()
        

        self.model = smp.Unet(
            encoder_name=encoder_name,      
            encoder_weights="imagenet" if pretrained else None, 
            in_channels=6,                   
            classes=1,                     
            activation='sigmoid'             
        )

    def forward(self, x1, x2):
    
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)