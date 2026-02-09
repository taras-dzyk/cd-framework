from models.baseline import SimpleCNN
from models.resnet_siam import ResNetSiamese
from models.unet_resnet import ResNetUNet
from models.hrnet import HRNetCD
from models.deeplab import DeepLabV3PlusCD

def get_model(model_name):
    name = model_name.lower()
    
    if name == "baseline":
        return SimpleCNN()
    
    elif name == "resnet18":
        return ResNetSiamese()
    
    elif name == "unet_resnet18":
        return ResNetUNet(encoder_name="resnet18")
    
    elif name == "hrnet":
        return HRNetCD(encoder_name="tu-hrnet_w18")
    
    elif name == "deeplab":
        return DeepLabV3PlusCD(encoder_name="resnet34")
        
    else:
        raise ValueError(f"Model {model_name} not found")