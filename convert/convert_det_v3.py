import os
import torch
import argparse
from collections import OrderedDict
import numpy as np
import time
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.utils.config import load_config
from convert.base_convert import BaseConvert, ONNXModel, TRTModel


class MobileV3DetConverter(BaseConvert):

    def __init__(self, config, device, paddle_pretrained_model_path,
                 distillation, **kwargs):
        self.distillation = distillation
        para_state_dict, _ = self.read_paddle_weights(
            paddle_pretrained_model_path)
        para_state_dict = self.del_invalid_state_dict(para_state_dict)
        super(MobileV3DetConverter, self).__init__(config, device, **kwargs)
        self.load_paddle_weights(para_state_dict)
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()

    def del_invalid_state_dict(self, para_state_dict):
        new_state_dict = OrderedDict()
        for i, (k, v) in enumerate(para_state_dict.items()):
            if k.startswith('Teacher.'):
                continue
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_paddle_weights(self, paddle_weights):
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
            if self.distillation:
                if k.startswith('backbone.') or k.startswith(
                        'neck.') or k.startswith('head.'):
                    ppname = 'Student.' + k
            ppname = ppname.replace('.running_mean', '._mean')
            ppname = ppname.replace('.running_var', '._variance')
            ppname = ppname.replace('.stages.', '.stage')
            # print(ppname)

            try:
                self.net.state_dict()[k].copy_(
                    torch.Tensor(para_state_dict[ppname]))
            except Exception as e:
                print('pytorch: {}, {}'.format(k, v.size()))
                print('paddle: {}, {}'.format(ppname,
                                              para_state_dict[ppname].shape))
                raise e

        print('model is loaded.')

    def pytorch2onnx(self, onnxPath):
        # dynamic_axes
        dynamic_axes = {'input': {0: 'batch_size'}}
        input_name = ['input']
        output_name = ['output']
        input = torch.randn(1, 3, 48, 320).to(self.device)
        torch.onnx.export(self.net,
                          input,
                          onnxPath,
                          opset_version=13,
                          input_names=input_name,
                          output_names=output_name,
                          verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        default=
        "/ldap_home/jingze.chi/PaddleOCR/pretrained/ch_PP-OCRv3_det_distill_train/best_accuracy",
        type=str)
    parser.add_argument("--distillation",
                        default=True,
                        type=str,
                        help="if weights are from distillation, set True")
    args = parser.parse_args()

    config = load_config('convert/det_v3_model_convert.yml')
    device = torch.device("cuda" if config['Global']['use_gpu'] else "cpu")
    converter = MobileV3DetConverter(config['Architecture'],
                                     device,
                                     args.weight_path,
                                     distillation=args.distillation)

    ## test data
    np.random.seed(666)
    inputs = np.random.randn(1, 3, 640, 640).astype(np.float32)
    inp = torch.from_numpy(inputs).to(device)

    ## summary
    from torchinfo import summary
    summary(converter.net, input_size=(1, 3, 640, 640), device="cuda:0")
    # out = converter.net(inp)
    # print(out.size())

    ## torch infer
    out_tr = converter.net(inp)
    # latency = []
    # for i in range(1000):
    #     t0 = time.time()
    #     out_tr = converter.net(inp)
    #     latency.append(time.time() - t0)
    # print("Average torch {} Inference time = {} ms".format(
    #     device.type, format(sum(latency) * 1000 / len(latency), '.2f')))
    out = out_tr['maps'].data.cpu().numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))
    # out_tr = out_tr.data.cpu().numpy()

    ## save torch model
    save_name = 'output/convert/det_v3.pth'
    converter.save_pytorch_weights(save_name)
    print('pytorch model save done.')
'''
    ## convert onnx and save
    onnx_path = 'output/convert/cn_model_onnx.onnx'
    converter.pytorch2onnx(onnx_path)
    print('onnx model save done.')

    ## test onnx
    onnxModel = ONNXModel(onnx_path, device)
    out_onnx = onnxModel(inputs)
    std = np.std(out_tr - out_onnx)
    print("std:", std)
    latency = []
    for i in range(1000):
        t0 = time.time()
        out_onnx = onnxModel(inputs)
        latency.append(time.time() - t0)
    print("Average onnx {} Inference time = {} ms".format(
        device.type, format(sum(latency) * 1000 / len(latency), '.2f')))

    ## onnx2tensorRT: /ldap_home/jingze.chi/TensorRT-8.2.5.1/bin/trtexec --onnx=model_onnx.onnx --verbose --explicitBatch --saveEngine=model_trt_fp16.engine --fp16
    tensorrt_path = 'output/convert/model_trt.engine'
    tensorrt_path_fp16 = 'output/convert/model_trt_fp16.engine'

    ## test trt
    trtModel = TRTModel(tensorrt_path, inputs.shape)
    out_trt = trtModel(inputs)
    std = np.std(out_tr - out_trt)
    print("std:", std)
    latency = []
    for i in range(1000):
        t0 = time.time()
        out_trt = trtModel(inputs)
        latency.append(time.time() - t0)
    print("Average trt {} Inference time = {} ms".format(
        device.type, format(sum(latency) * 1000 / len(latency), '.2f')))

    ## test trt fp16
    trtModel = TRTModel(tensorrt_path_fp16, inputs.shape)
    out_trt = trtModel(inputs)
    std = np.std(out_tr - out_trt)
    print("std:", std)
    latency = []
    for i in range(1000):
        t0 = time.time()
        out_trt = trtModel(inputs)
        latency.append(time.time() - t0)
    print("Average trtfp16 {} Inference time = {} ms".format(
        device.type, format(sum(latency) * 1000 / len(latency), '.2f')))
'''