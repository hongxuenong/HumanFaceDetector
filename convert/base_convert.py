import os
import sys
import torch
import numpy as np
import tensorrt as trt
import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit  # don't delete if want to us pycuda

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.modeling.architectures import build_model


class BaseConvert:

    def __init__(self, config, device, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()
        self.device = device
        self.net.to(self.device)

    def build_net(self, **kwargs):
        self.net = build_model(self.config, **kwargs)

    def load_paddle_weights(self, weights_path):
        raise NotImplementedError

    def pytorch2onnx(self, onnxPath):
        raise NotImplementedError

    def pytorch2trt(self, trtPath):
        raise NotImplementedError

    def read_pytorch_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = torch.load(weights_path, map_location=self.device)
        return weights

    def get_out_channels(self, weights):
        if list(weights.keys())[-1].endswith('.weight') and len(
                list(weights.values())[-1].shape) == 2:
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        print('weighs is loaded.')

    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path))
        print('model is loaded: {}'.format(weights_path))

    def save_pytorch_weights(self, weights_path):
        torch.save(
            self.net.state_dict(), weights_path
        )  #  for torch>=1.6.0 file format will be a new zip format, _use_new_zipfile_serialization=False to change it
        print('model is saved: {}'.format(weights_path))

    def print_pytorch_state_dict(self):
        print('pytorch:')
        for k, v in self.net.state_dict().items():
            print('{}----{}'.format(k, type(v)))

    def read_paddle_weights(self, weights_path):
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        return para_state_dict, opti_state_dict

    def print_paddle_state_dict(self, weights_path):
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        print('paddle"')
        for k, v in para_state_dict.items():
            print('{}----{}'.format(k, type(v)))

    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer


class ONNXModel():

    def __init__(self, onnx_path, device):
        """
        :param onnx_path:
        """
        self.device = device
        if self.device.type == 'cuda':
            provider = ['CUDAExecutionProvider']
        else:
            provider = ['CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(onnx_path,
                                                         providers=provider)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def __call__(self, image_numpy):
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        outputs = self.onnx_session.run(self.output_name,
                                        input_feed=input_feed)
        return outputs


class TRTModel():

    def __init__(self, trt_path, inshape, target_dtype=np.float32):
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        with open(trt_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.outshape = self.context.get_binding_shape(1)

        input_batch = np.random.randn(*inshape).astype(target_dtype)
        self.output = np.empty(self.outshape, dtype=target_dtype)

        self.d_input = cuda.mem_alloc(1 * input_batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def __call__(self, batch):  # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # execute model, Synchronous infer: execute_async_v2->execute_v2
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # syncronize threads
        self.stream.synchronize()
        return self.output