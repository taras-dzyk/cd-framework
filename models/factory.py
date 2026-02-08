from models.baseline import SimpleCNN
from models.resnet_siam import ResNetSiamese

def get_model(model_name):
    models_map = {
        "baseline": SimpleCNN,
        "resnet18": ResNetSiamese,
        # Сюди легко додати нові: "resnet34": ResNet34Siamese
    }
    
    if model_name not in models_map:
        raise ValueError(f"Модель {model_name} не підтримується")
        
    return models_map[model_name]()