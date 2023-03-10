import torch
from torch import reshape, transpose
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Linear, Dropout
from torch.nn import AdaptiveAvgPool2d, MaxPool2d, AvgPool2d
from torch.utils.checkpoint import checkpoint
import torch.nn.init as init


class ConvBNLayer(nn.Module):

    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2d(in_channels=num_channels,
                            out_channels=num_filters,
                            kernel_size=filter_size,
                            stride=stride,
                            padding=padding,
                            groups=num_groups,
                            bias=False)

        self._batch_norm = BatchNorm2d(num_filters)
        # param_attr=ParamAttr(regularizer=L2Decay(0.0))?
        # bias_attr=ParamAttr(regularizer=L2Decay(0.0))?

        # hard_swish activation
        self._hardswish = nn.Hardswish()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                init.kaiming_normal_(m.bias.data)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = self._hardswish(x)
        return x


class DepthwiseSeparable(nn.Module):

    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale,
                 dw_size=3,
                 padding=1,
                 use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self._depthwise_conv = ConvBNLayer(num_channels=num_channels,
                                           num_filters=int(num_filters1 *
                                                           scale),
                                           filter_size=dw_size,
                                           stride=stride,
                                           padding=padding,
                                           num_groups=int(num_groups * scale))
        if use_se:
            self._se = SEModule(int(num_filters1 * scale))
        self._pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, x):
        x = self._depthwise_conv(x)
        if self.use_se:
            x = self._se(x)
        x = self._pointwise_conv(x)
        return x


class MobileNetV1Enhance(nn.Module):

    def __init__(self,
                 in_channels=3,
                 scale=0.5,
                 last_conv_stride=1,
                 last_pool_type='max',
                 **kwargs):
        super().__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(num_channels=3,
                                 filter_size=3,
                                 num_filters=int(32 * scale),
                                 stride=2,
                                 padding=1)

        conv2_1 = DepthwiseSeparable(num_channels=int(32 * scale),
                                     num_filters1=32,
                                     num_filters2=64,
                                     num_groups=32,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(num_channels=int(64 * scale),
                                     num_filters1=64,
                                     num_filters2=128,
                                     num_groups=64,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(num_channels=int(128 * scale),
                                     num_filters1=128,
                                     num_filters2=128,
                                     num_groups=128,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(num_channels=int(128 * scale),
                                     num_filters1=128,
                                     num_filters2=256,
                                     num_groups=128,
                                     stride=(2, 1),
                                     scale=scale)
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(num_channels=int(256 * scale),
                                     num_filters1=256,
                                     num_filters2=256,
                                     num_groups=256,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(num_channels=int(256 * scale),
                                     num_filters1=256,
                                     num_filters2=512,
                                     num_groups=256,
                                     stride=(2, 1),
                                     scale=scale)
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthwiseSeparable(num_channels=int(512 * scale),
                                       num_filters1=512,
                                       num_filters2=512,
                                       num_groups=512,
                                       stride=1,
                                       dw_size=5,
                                       padding=2,
                                       scale=scale,
                                       use_se=False)
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparable(num_channels=int(512 * scale),
                                     num_filters1=512,
                                     num_filters2=1024,
                                     num_groups=512,
                                     stride=(2, 1),
                                     dw_size=5,
                                     padding=2,
                                     scale=scale,
                                     use_se=True)
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparable(num_channels=int(1024 * scale),
                                   num_filters1=1024,
                                   num_filters2=1024,
                                   num_groups=1024,
                                   stride=last_conv_stride,
                                   dw_size=5,
                                   padding=2,
                                   use_se=True,
                                   scale=scale)
        self.block_list.append(conv6)

        self.block_list = nn.Sequential(*self.block_list)
        if last_pool_type == 'avg':
            self.pool = AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

        self.use_checkpoint = kwargs.get('use_checkpoint', False)

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint(self.conv1,x)
            x = checkpoint(self.block_list,x)
            x = checkpoint(self.pool,x)
        else:
            x = self.conv1(x)
            x = self.block_list(x)
            x = self.pool(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(in_channels=channel,
                            out_channels=channel // reduction,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(in_channels=channel // reduction,
                            out_channels=channel,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hardsigmoid(outputs)
        return torch.mul(inputs, outputs)


if __name__ == "__main__":

    from torchinfo import summary

    arr = torch.rand((1, 3, 48, 320))
    model = MobileNetV1Enhance()
    summary(model, input_size=(1, 3, 48, 320), device="cpu")
    out = model(arr)
    print(out.size())
