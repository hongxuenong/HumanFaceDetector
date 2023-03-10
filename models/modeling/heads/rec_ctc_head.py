

import math

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

# def get_para_bias_attr(l2_decay, k):
#     regularizer = paddle.regularizer.L2Decay(l2_decay)
#     stdv = 1.0 / math.sqrt(k * 1.0)
#     initializer = nn.initializer.Uniform(-stdv, stdv)
#     weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
#     bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
#     return [weight_attr, bias_attr]


class CTCHead(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            # weight_attr, bias_attr = get_para_bias_attr(
            #     l2_decay=fc_decay, k=in_channels)
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                # weight_attr=weight_attr,
                # bias_attr=bias_attr
            )
        else:
            # weight_attr1, bias_attr1 = get_para_bias_attr(
            #     l2_decay=fc_decay, k=in_channels)
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                # weight_attr=weight_attr1,
                # bias_attr=bias_attr1
            )

            # weight_attr2, bias_attr2 = get_para_bias_attr(
            #     l2_decay=fc_decay, k=mid_channels)
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                # weight_attr=weight_attr2,
                # bias_attr=bias_attr2
            )
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.in_channels = in_channels

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(self.in_channels * 1.0)
            init.uniform_(m.weight.data, -stdv, stdv)
            if m.bias is not None:
                init.uniform_(m.bias.data, -stdv, stdv)
                
    def forward(self, x, targets=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result

if __name__=="__main__":

    from torchinfo import summary

    arr = torch.rand((1,80,64))
    model = CTCHead(64, 3000)
    summary(model, input_size=(1, 80, 64), device="cpu")
    out = model(arr)
    print(out.size())
    # print(model)

