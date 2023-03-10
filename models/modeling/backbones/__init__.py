__all__ = ["build_backbone"]

from .det_mobilenet_v3 import MobileNetV3
from .det_resnet_vd import ResNet
from .swin import SwinTransformer
from .efficientnet import EfficientNet
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import mobilenet_v3_large

def build_backbone(config):
    module_name = config.pop("name")
    module_class = eval(module_name)(**config)
    return module_class
