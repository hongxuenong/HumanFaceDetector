# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.transformer import build_norm_layer, build_activation
from ..utils.weight_init import kaiming_init, constant_init


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()

        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            self.padding_layer = nn.ZeroPad2d(padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=conv_padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            # if self.with_bias:
            #     if isinstance(norm, (_BatchNorm, _InstanceNorm)):
            #         warnings.warn(
            #             'Unnecessary conv bias before batch/instance norm')
        else:
            self.norm_name = None

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


class FPNC(nn.Module):
    """FPN-like fusion module in Real-time Scene Text Detection with
    Differentiable Binarization.

    This was partially adapted from https://github.com/MhLiao/DB and
    https://github.com/WenmuZhou/DBNet.pytorch.

    Args:
        in_channels (list[int]): A list of numbers of input channels.
        lateral_channels (int): Number of channels for lateral layers.
        out_channels (int): Number of output channels.
        bias_on_lateral (bool): Whether to use bias on lateral convolutional
            layers.
        bn_re_on_lateral (bool): Whether to use BatchNorm and ReLU
            on lateral convolutional layers.
        bias_on_smooth (bool): Whether to use bias on smoothing layer.
        bn_re_on_smooth (bool): Whether to use BatchNorm and ReLU on smoothing
            layer.
        asf_cfg (dict): Adaptive Scale Fusion module configs. The
            attention_type can be 'ScaleChannelSpatial'.
        conv_after_concat (bool): Whether to add a convolution layer after
            the concatenation of predictions.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 lateral_channels=256,
                 out_channels=64,
                 upsample='bilinear',
                 bias_on_lateral=False,
                 bn_re_on_lateral=False,
                 bias_on_smooth=False,
                 bn_re_on_smooth=False,
                 asf_cfg=None,
                 conv_after_concat=False):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.lateral_channels = lateral_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.bn_re_on_lateral = bn_re_on_lateral
        self.bn_re_on_smooth = bn_re_on_smooth
        self.asf_cfg = asf_cfg
        self.conv_after_concat = conv_after_concat
        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()
        self.num_outs = self.num_ins
        self.upsample = upsample
        for i in range(self.num_ins):
            norm_cfg = None
            act_cfg = None
            if self.bn_re_on_lateral:
                norm_cfg = dict(type='BN')
                act_cfg = dict(type='ReLU')
            l_conv = ConvModule(in_channels[i],
                                lateral_channels,
                                1,
                                bias=bias_on_lateral,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg,
                                inplace=False)
            norm_cfg = None
            act_cfg = None
            if self.bn_re_on_smooth:
                norm_cfg = dict(type='BN')
                act_cfg = dict(type='ReLU')

            smooth_conv = ConvModule(lateral_channels,
                                     out_channels,
                                     3,
                                     bias=bias_on_smooth,
                                     padding=1,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg,
                                     inplace=False)

            self.lateral_convs.append(l_conv)
            self.smooth_convs.append(smooth_conv)

        if self.asf_cfg is not None:
            self.asf_conv = ConvModule(out_channels * self.num_outs,
                                       out_channels * self.num_outs,
                                       3,
                                       padding=1,
                                       norm_cfg=None,
                                       act_cfg=None,
                                       inplace=False)
            if self.asf_cfg['attention_type'] == 'ScaleChannelSpatial':
                self.asf_attn = ScaleChannelSpatialAttention(
                    self.out_channels * self.num_outs,
                    (self.out_channels * self.num_outs) // 4, self.num_outs)
            else:
                raise NotImplementedError

        if self.conv_after_concat:
            norm_cfg = dict(type='BN')
            act_cfg = dict(type='ReLU')
            self.out_conv = ConvModule(out_channels * self.num_outs,
                                       out_channels * self.num_outs,
                                       3,
                                       padding=1,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg,
                                       inplace=False)

    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Each tensor has the shape of
                :math:`(N, C_i, H_i, W_i)`. It usually expects 4 tensors
                (C2-C5 features) from ResNet.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H_0, W_0)` where
            :math:`C_{out}` is ``out_channels``.
        """
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        # build top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode=self.upsample)
        # build outputs
        # part 1: from original levels
        outs = [
            self.smooth_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        for i, out in enumerate(outs):
            outs[i] = F.interpolate(outs[i],
                                    size=outs[0].shape[2:],
                                    mode=self.upsample)

        out = torch.cat(outs, dim=1)
        if self.asf_cfg is not None:
            asf_feature = self.asf_conv(out)
            attention = self.asf_attn(asf_feature)
            enhanced_feature = []
            for i, out in enumerate(outs):
                enhanced_feature.append(attention[:, i:i + 1] * outs[i])
            out = torch.cat(enhanced_feature, dim=1)

        if self.conv_after_concat:
            out = self.out_conv(out)

        return out


class ScaleChannelSpatialAttention(nn.Module):
    """Spatial Attention module in Real-Time Scene Text Detection with
    Differentiable Binarization and Adaptive Scale Fusion.

    This was partially adapted from https://github.com/MhLiao/DB

    Args:
        in_channels (int): A numbers of input channels.
        c_wise_channels (int): Number of channel-wise attention channels.
        out_channels (int): Number of output channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 c_wise_channels,
                 out_channels,
                 init_cfg=[dict(type='Kaiming', layer='Conv', bias=0)]):
        super().__init__(init_cfg=init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Channel Wise
        self.channel_wise = nn.Sequential(
            ConvModule(in_channels,
                       c_wise_channels,
                       1,
                       bias=False,
                       norm_cfg=None,
                       act_cfg=dict(type='ReLU'),
                       inplace=False),
            ConvModule(c_wise_channels,
                       in_channels,
                       1,
                       bias=False,
                       norm_cfg=None,
                       act_cfg=dict(type='Sigmoid'),
                       inplace=False))
        # Spatial Wise
        self.spatial_wise = nn.Sequential(
            ConvModule(1,
                       1,
                       3,
                       padding=1,
                       bias=False,
                       norm_cfg=None,
                       act_cfg=dict(type='ReLU'),
                       inplace=False),
            ConvModule(1,
                       1,
                       1,
                       bias=False,
                       norm_cfg=None,
                       act_cfg=dict(type='Sigmoid'),
                       inplace=False))
        # Attention Wise
        self.attention_wise = ConvModule(in_channels,
                                         out_channels,
                                         1,
                                         bias=False,
                                         norm_cfg=None,
                                         act_cfg=dict(type='Sigmoid'),
                                         inplace=False)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): A concat FPN feature tensor that has the shape of
                :math:`(N, C, H, W)`.

        Returns:
            Tensor: An attention map of shape :math:`(N, C_{out}, H, W)`
            where :math:`C_{out}` is ``out_channels``.
        """
        out = self.avg_pool(inputs)
        out = self.channel_wise(out)
        out = out + inputs
        inputs = torch.mean(out, dim=1, keepdim=True)
        out = self.spatial_wise(inputs) + out
        out = self.attention_wise(out)

        return out
