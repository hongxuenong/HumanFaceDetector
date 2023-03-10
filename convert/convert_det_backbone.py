import os
from unittest import mock
import torch
import argparse
from collections import OrderedDict
import numpy as np
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.utils.config import load_config
from convert.base_convert import BaseConvert, ONNXModel, TRTModel


class DetBackboneConverter(BaseConvert):

    def __init__(self, config, device, paddle_pretrained_model_path,
                 model_type, **kwargs):
        para_state_dict, _ = self.read_paddle_weights(
            paddle_pretrained_model_path)
        super(DetBackboneConverter, self).__init__(config, device, **kwargs)
        self.load_paddle_weights(para_state_dict, model_type)
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()

    def load_paddle_weights(self, paddle_weights, model_type):
        para_state_dict = paddle_weights
        for per in para_state_dict:
            print(per)
        print("==========================")
        for per, v in self.net.state_dict().items():
            print(per)
        print("==========================")
        for k, v in self.net.state_dict().items():
            if k.endswith('num_batches_tracked'):
                continue

            ppname = k

            ppname = ppname.replace('.running_mean', '._mean')
            ppname = ppname.replace('.running_var', '._variance')
            if "stages" in ppname:
                if model_type == "resnet":
                    import re
                    ppname = re.sub(r'(?<=stages\.[0-9])\.', "_", ppname)
                    ppname = ppname.replace(
                        'stages.',
                        'bb_')  # backbone.stages.0.0 -> backbone.bb_0_0
                elif model_type == "mobilenetv3":
                    ppname = ppname.replace(
                        '.stages.',
                        '.stage')  # backbone.stages.0.0 -> backbone.stage0.0

            try:
                self.net.state_dict()[k].copy_(
                    torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname,
                                              para_state_dict[ppname].shape))
                raise e

        print('model is loaded.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        default="pretrain_models/MobileNetV3_large_x0_5_pretrained.pdparams",
        type=str)
    args = parser.parse_args()

    config = load_config('convert/mobilenetv3_model_convert.yml')
    device = torch.device("cuda" if config['Global']['use_gpu'] else "cpu")
    converter = DetBackboneConverter(config['Architecture'],
                                     device,
                                     args.weight_path,
                                     model_type="mobilenetv3")

    ## test data
    np.random.seed(666)
    inputs = np.random.randn(1, 3, 640, 640).astype(np.float32)
    inp = torch.from_numpy(inputs).to(device)

    ## summary
    from torchinfo import summary
    summary(converter.net, input_size=(1, 3, 640, 640), device="cuda:0")

    ## torch infer
    # out_tr = converter.net(inp)
    # out = out_tr.data.cpu().numpy()
    # print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    ## save torch model
    save_name = 'output/convert/mobilenetv3_large_x0_5.pth'  #'output/convert/resnet50_vd_ssld.pth'
    converter.save_pytorch_weights(save_name)
    print('pytorch model save done.')