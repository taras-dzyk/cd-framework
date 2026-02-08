import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Гілка для екстракції ознак (однакова для A та B)
        # Використовуємо послідовність: Conv -> BatchNorm -> ReLU -> Pool
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 256 -> 128
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128 -> 64
        )
        
        # Декодер: приймає об'єднані ознаки та відновлює маску
        # Після конкатенації у нас буде 32 + 32 = 64 канали
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 64 -> 128
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 128 -> 256
            
            nn.Conv2d(16, 1, kernel_size=1), # Фінальний шар: 1 канал для маски
            nn.Sigmoid() # Щоб отримати значення від 0 до 1
        )

    def forward(self, x1, x2):
        # Пропускаємо обидва зображення через одну і ту ж мережу
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)
        
        # Конкатенація по виміру каналів (dim=1)
        combined = torch.cat((feat1, feat2), dim=1)
        
        # Отримуємо фінальну маску
        mask = self.decoder(combined)
        return mask

if __name__ == "__main__":
    # Тест форми тензорів
    model = SimpleCNN()
    x1 = torch.randn(1, 3, 256, 256)
    x2 = torch.randn(1, 3, 256, 256)
    output = model(x1, x2)
    print(f"Вхід: {x1.shape}")
    print(f"Вихід (маска): {output.shape}") # Має бути [1, 1, 256, 256]