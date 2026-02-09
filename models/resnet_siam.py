import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import ssl
import os
os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

class ResNetSiamese(nn.Module):
    def __init__(self):
        super(ResNetSiamese, self).__init__()
        
       
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,   # 1/4 розміру
            resnet.layer1,    # 1/4 розміру
            resnet.layer2,    # 1/8 розміру
            resnet.layer3,    # 1/16 розміру
        )
        
        # Декодер для відновлення маски
        # На виході layer3 ми маємо 256 каналів. 
        # Оскільки у нас 2 гілки, після конкатенації буде 512.
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), # Повертаємо 1/2
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # Повертаємо в 1/1
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        
        combined = torch.cat((feat1, feat2), dim=1)
        mask = self.decoder(combined)
        return mask

if __name__ == "__main__":
    model = ResNetSiamese()
    x = torch.randn(1, 3, 256, 256)
    out = model(x, x)
    print(f"ResNet-Siam output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params:,}")