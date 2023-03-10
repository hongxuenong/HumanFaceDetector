import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ['MobileNetV3']


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Module):

    def __init__(self,
                 in_channels=3,
                 model_name='large',
                 scale=0.5,
                 disable_se=False,
                 **kwargs):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super(MobileNetV3, self).__init__()

        self.disable_se = disable_se

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 2],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', 2],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', 2],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', 2],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        inplanes = 16
        # conv1
        self.conv = ConvBNLayer(in_channels=in_channels,
                                out_channels=make_divisible(inplanes * scale),
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=1,
                                if_act=True,
                                act='hardswish')

        self.stages = nn.ModuleList()
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            se = se and not self.disable_se
            start_idx = 2 if model_name == 'large' else 0
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(
                ResidualUnit(in_channels=inplanes,
                             mid_channels=make_divisible(scale * exp),
                             out_channels=make_divisible(scale * c),
                             kernel_size=k,
                             stride=s,
                             use_se=se,
                             act=nl))
            inplanes = make_divisible(scale * c)
            i += 1
        block_list.append(
            ConvBNLayer(in_channels=inplanes,
                        out_channels=make_divisible(scale * cls_ch_squeeze),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=1,
                        if_act=True,
                        act='hardswish'))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))
        # for i, stage in enumerate(self.stages):
        #     self.add_module("stage{}".format(i), stage)
        self.use_checkpoint = kwargs.get('use_checkpoint', False)

    def forward(self, x):
        if self.use_checkpoint:
            x = checkpoint(self.conv, x)
            out_list = []
            for stage in self.stages:
                x = checkpoint(stage, x)
                out_list.append(x)
        else:
            x = self.conv(x)
            out_list = []
            for stage in self.stages:
                x = stage(x)
                out_list.append(x)
        return out_list


class ConvBNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x, inplace=True)
            elif self.act == "hardswish":
                x = F.hardswish(x, inplace=True)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


class ResidualUnit(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(in_channels=in_channels,
                                       out_channels=mid_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       if_act=True,
                                       act=act)
        self.bottleneck_conv = ConvBNLayer(in_channels=mid_channels,
                                           out_channels=mid_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=int((kernel_size - 1) // 2),
                                           groups=mid_channels,
                                           if_act=True,
                                           act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(in_channels=mid_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       if_act=False,
                                       act=None)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x += inputs  #bug
        return x


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction,
                               out_channels=in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.hard_sigmoid = Hsigmoid(inplace=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs, inplace=True)
        outputs = self.conv2(outputs)
        # outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        outputs = self.hard_sigmoid(outputs)
        return inputs * outputs


if __name__ == "__main__":

    # from torchinfo import summary
    model = MobileNetV3(in_channels=3, disable_se=True)
    # summary(model, input_size=(1, 3, 640, 640), device="cuda:1")
    # for key in model.state_dict().keys():
        # print(key)
    # print(len(model.state_dict().keys()))
    arr = torch.rand((8, 3, 640, 640))
    out = model(arr)
    print([o.shape for o in out])