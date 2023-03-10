import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
import math

# def get_bias_attr(k):
#     stdv = 1.0 / math.sqrt(k * 1.0)
#     initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
#     bias_attr = ParamAttr(initializer=initializer)
#     return bias_attr


class Head(nn.Module):

    def __init__(self,
                 in_channels,
                 name_list,
                 kernel_list=[3, 2, 2],
                 **kwargs):
        super(Head, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            # weight_attr=ParamAttr(),
            bias=False)
        self.conv_bn1 = nn.BatchNorm2d(
            in_channels // 4,
            # param_attr=ParamAttr(
            #     initializer=paddle.nn.initializer.Constant(value=1.0)),
            # bias_attr=ParamAttr(
            #     initializer=paddle.nn.initializer.Constant(value=1e-4)),
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            # weight_attr=ParamAttr(
            #     initializer=paddle.nn.initializer.KaimingUniform()),
            # bias_attr=get_bias_attr(in_channels // 4)
        )
        self.conv_bn2 = nn.BatchNorm2d(
            in_channels // 4,
            # param_attr=ParamAttr(
            #     initializer=paddle.nn.initializer.Constant(value=1.0)),
            # bias_attr=ParamAttr(
            #     initializer=paddle.nn.initializer.Constant(value=1e-4)),
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.ConvTranspose2d(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            # weight_attr=ParamAttr(
            #     initializer=paddle.nn.initializer.KaimingUniform()),
            # bias_attr=get_bias_attr(in_channels // 4),
        )
        self.in_channels = in_channels
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                stdv = 1.0 / math.sqrt((self.in_channels // 4) * 1.0)
                init.uniform_(m.bias.data, -stdv, stdv)
        elif isinstance(m, nn.BatchNorm2d):
            # init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 1e-4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        binarize_name_list = [
            'conv2d_56', 'batch_norm_47', 'conv2d_transpose_0',
            'batch_norm_48', 'conv2d_transpose_1', 'binarize'
        ]
        thresh_name_list = [
            'conv2d_57', 'batch_norm_49', 'conv2d_transpose_2',
            'batch_norm_50', 'conv2d_transpose_3', 'thresh'
        ]
        self.binarize = Head(in_channels, binarize_name_list, **kwargs)
        self.thresh = Head(in_channels, thresh_name_list, **kwargs)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], axis=1)
        return {'maps': y}


if __name__ == "__main__":

    # from torchinfo import summary
    model = DBHead(in_channels=256).cuda()
    # summary(model, input_size=(1, 512, 20, 20), device="cuda:1")

    arr = torch.rand((8, 256, 20, 20)).cuda()
    out = model(arr)
