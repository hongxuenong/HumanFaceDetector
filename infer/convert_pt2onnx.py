import os
import sys
import torch
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.utils.config import add_config_rec, load_config
from convert.base_convert import BaseConvert, ONNXModel
from models.utils.utility import get_char_num


class Pt2Onnx(BaseConvert):

    def __init__(self,
                 config,
                 device,
                 model_path,
                 model_type,
                 distillation=True,
                 **kwargs):
        self.device = device
        self.model_type = model_type
        self.distillation = distillation
        super(Pt2Onnx, self).__init__(config, device, **kwargs)
        weights = self.read_pytorch_weights(model_path)
        self.load_state_dict(weights)
        self.net.eval()

    def pytorch2onnx(self, onnxPath):
        if self.model_type == "det":
            self.pytorch2onnx_det(onnxPath)
        elif self.model_type == "rec":
            self.pytorch2onnx_rec(onnxPath)
        else:
            raise Exception("model type is not supported!")

    def pytorch2onnx_rec(self, onnxPath):
        # dynamic_axes
        dynamic_axes = {'input': {0: 'batch_size', 3: 'width'}}
        input_name = ['input']
        output_name = ['output']
        input = torch.randn(1, 3, 48, 320).to(self.device)
        torch.onnx.export(self.net,
                          input,
                          onnxPath,
                          opset_version=13,
                          input_names=input_name,
                          output_names=output_name,
                          dynamic_axes=dynamic_axes,
                          verbose=True)

    def pytorch2onnx_det(self, onnxPath):
        # dynamic_axes
        dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'}}
        input_name = ['input']
        output_name = ['output']
        input = torch.randn(1, 3, 640, 640).to(self.device)
        torch.onnx.export(self.net,
                          input,
                          onnxPath,
                          opset_version=13,
                          input_names=input_name,
                          output_names=output_name,
                          dynamic_axes=dynamic_axes,
                          verbose=True)

    def load_state_dict(self, weights):
        if self.distillation:
            new_weights = {}
            # print(weights)
            for key in weights:
                
                if "Student." in key:
                    new_weights[key.replace("Student.",
                                            "").replace("model_dict.",
                                                        "")] = weights[key]
            
            # print(new_weights)
            self.net.load_state_dict(new_weights)
        else:
            self.net.load_state_dict(weights)
        print('weighs is loaded.')


if __name__ == "__main__":
    model_type = "det"  # det or rec
    weight_path = "output/det_swin2mbv3_feat_dist_cml_v2/best_accuracy.ptparams"
    config = load_config('configs/det/det_swin_db_mobilenetv3.yml')

    ## for rec
    if model_type == "rec":
        char_num = get_char_num(
            "models/rec/rec_v3_mobile_dis_vn/char_VN.txt") + len(['blank'])
        config = add_config_rec(config, char_num)

    device = torch.device("cuda" if config['Global']['use_gpu'] else "cpu")

    converter = Pt2Onnx(config['Architecture'], device, weight_path,
                        model_type)

    ## convert onnx and save
    onnx_path = "models/det/dbnet_swinbase/det.onnx"
    converter.pytorch2onnx(onnx_path)
    print('onnx model save done.')

    ## test data
    np.random.seed(666)

    inputs = np.random.randn(1, 3, 48, 340).astype(np.float32)
    if model_type == "det":
        inputs = np.random.randn(1, 3, 640, 640).astype(np.float32)

    inp = torch.from_numpy(inputs).to(device)

    ## torch infer
    out_tr = converter.net(inp)
    if model_type == "det":
        out_tr = out_tr["maps"]
    out_tr = out_tr.data.cpu().numpy()

    ## test onnx
    onnxModel = ONNXModel(onnx_path, device)
    out_onnx = onnxModel(inputs)
    std = np.std(out_tr - out_onnx)
    print("std:", std)
