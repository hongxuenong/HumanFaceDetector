from torch import nn
# from ..transforms import build_transform
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head
from .base_model import BaseModel
from ...utils.save_load import load_pretrained_params
import math

__all__ = ['DistillationModel']


class DistillationModel(nn.Module):
    def __init__(self, config):
        """
        the module for OCR distillation.
        args:
            config (dict): the super parameters for module.
        """
        super().__init__()
        self.model_list = []
        self.model_dict = nn.ModuleDict()
        self.model_name_list = []
        for key in config["Models"]:
            model_config = config["Models"][key]
            freeze_params = False
            pretrained = None
            if "freeze_params" in model_config:
                freeze_params = model_config.pop("freeze_params")
            if "pretrained" in model_config:
                pretrained = model_config.pop("pretrained")
            model = BaseModel(model_config)
            if pretrained is not None:
                load_pretrained_params(model, pretrained)
            if freeze_params:
                for param in model.parameters():
                    param.requires_grad = False
            # self.model_list.append(self.add_module(key, model))
            # self.add_module(key, model) #0
            self.model_dict[key] = model #1
            self.model_name_list.append(key)
        
        self.Connectors = None
        if 'Connector' in config:
            self.Connectors = nn.ModuleList([self.build_feature_connector(t, s) for t, s in zip(config['Connector']['t_channels'], config['Connector']['s_channels'])])
            # self.Connectors = self.Connectors.to('cuda')

        # check trainable params
        # for name, param in self.model_dict['Student'].named_parameters():
        #     if param.requires_grad:
        #         print(name)
        
    def build_feature_connector(self, t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(t_channel)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, x, data=None):
        result_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            # result_dict[model_name] = self.model_list[idx](x, data)
            # result_dict[model_name] = self.get_submodule(model_name)(x, data) #0
            result_dict[model_name] = self.model_dict[model_name](x, data) #1
        
        if self.Connectors is not None and self.training:
            feat_num = len(result_dict['Teacher']['backbone_out'])
            stu_feats = result_dict['Student']['backbone_out']
            stu2_feats = result_dict['Student2']['backbone_out']
            for i in range(feat_num):
                stu_feats[i] = self.Connectors[i](stu_feats[i])
                stu2_feats[i] = self.Connectors[i](stu2_feats[i])
            result_dict['Student_Connector'] = {}
            result_dict['Student2_Connector'] = {}
            result_dict['Student_Connector']['backbone_out'] = stu_feats
            result_dict['Student2_Connector']['backbone_out'] = stu2_feats
        return result_dict