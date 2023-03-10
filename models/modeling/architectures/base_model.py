import torch
from torch import nn
from ..backbones import build_backbone
from ..necks import build_neck
from ..heads import build_head
from ..transforms import build_transform


class BaseModel(nn.Module):

    def __init__(self, config):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()
        in_channels = config.get('in_channels', 3)
        self.model_type = config['model_type']
        # build transfrom,
        # for rec, transfrom can be TPS,None
        # for det and cls, transfrom shoule to be None,
        # if you make model differently, you can use transfrom in det and cls
        if 'Transform' not in config or config['Transform'] is None:
            self.use_transform = False
        else:
            self.use_transform = True
            config['Transform']['in_channels'] = in_channels
            self.transform = build_transform(config['Transform'])
            in_channels = self.transform.out_channels

        # build backbone, backbone is need for det, rec and cls
        config["Backbone"]['in_channels'] = in_channels
        self.backbone = build_backbone(config["Backbone"], self.model_type)
        in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if 'Neck' not in config or config['Neck'] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            freeze_params = False
            if "freeze_params" in config['Neck']:
                freeze_params = config['Neck'].pop("freeze_params")
            config['Neck']['in_channels'] = in_channels
            self.neck = build_neck(config['Neck'])
            in_channels = self.neck.out_channels
            
            if freeze_params:
                for param in self.neck.parameters():
                    param.requires_grad = False

        # # build head, head is need for det, rec and cls
        if 'Head' not in config or config['Head'] is None:
            self.use_head = False
        else:
            self.use_head = True
            freeze_params = False
            if "freeze_params" in config['Head']:
                freeze_params = config['Head'].pop("freeze_params")
            if 'in_channels' not in config[
                    "Head"] or config["Head"]['in_channels'] is None:
                config["Head"]['in_channels'] = in_channels
            self.head = build_head(config["Head"])
            
            if freeze_params:
                for param in self.head.parameters():
                    param.requires_grad = False

        self.return_all_feats = config.get("return_all_feats", False)

    def forward(self, x, data=None):
        y = dict()
        if self.use_transform:
            x = self.transform(x)
        x = self.backbone(x)
        y["backbone_out"] = x
        if self.use_neck:
            x = self.neck(x)
        y["neck_out"] = x
        if self.use_head:
            x = self.head(x, targets=data)
        # for multi head, save ctc neck out for udml
        if isinstance(x, dict) and 'ctc_neck' in x.keys():
            y["neck_out"] = x["ctc_neck"]
            y["head_out"] = x
        elif isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        if self.return_all_feats:
            if self.training:
                return y
            else:
                if self.model_type == 'det':
                    return {"maps": y["maps"]}
                else:
                    return {"head_out": y["head_out"]}
        else:
            return x
