import numpy as np
import time
from utility import ONNXModel, TRTModel

if __name__ == "__main__":
    ## rec
    # /ldap_home/jingze.chi/TensorRT-8.2.5.1/bin/trtexec \
    # --onnx=models/rec/rec_v3_mobile_dis_cn/cn.onnx \
    # --verbose --explicitBatch \
    # --minShapes=input:1x3x48x320 \
    #  --optShapes=input:1x3x48x320 \
    # --maxShapes=input:16x3x48x2000 \
    # --saveEngine=models/rec/rec_v3_mobile_dis_cn/cn_fp16.trt \
    # --fp16

    ## det
    # /ldap_home/jingze.chi/TensorRT-8.2.5.1/bin/trtexec \
    # --onnx=models/det/MobileNetV3_cml_weight/det_mobile.onnx \
    # --verbose --explicitBatch \
    # --minShapes=input:1x3x32x32 \
    # --optShapes=input:1x3x960x960 \
    # --maxShapes=input:8x3x1280x1280 \
    # --saveEngine=models/det/MobileNetV3_cml_weight/det_mobile_fp16.trt \
    # --fp16

    tensorrt_path = 'models/rec/rec_v3_mobile_dis_cn/cn_fp32.trt'
    tensorrt_path_fp16 = 'models/rec/rec_v3_mobile_dis_cn/cn_fp16.trt'

    ## test data
    batch_size = 8
    np.random.seed(666)
    inputs = np.random.randn(batch_size, 3, 48, 320).astype(np.float32)

    ## test onnx
    onnxModel = ONNXModel('models/rec/rec_v3_mobile_dis_cn/cn.onnx',
                          use_gpu=True)
    out_onnx = onnxModel(inputs)[0]
    latency = []
    for i in range(1000):
        t0 = time.time()
        out_onnx_t = onnxModel(inputs)
        latency.append(time.time() - t0)
    print("Average onnx Inference time = {} ms".format(
        format(sum(latency) * 1000 / len(latency), '.2f')))

    ## test trt
    trtModel = TRTModel(tensorrt_path, (batch_size, 3, 48, 2000))
    out_trt = trtModel(inputs)[0]
    std = np.std(out_onnx - out_trt)
    print("std:", std)
    latency = []
    for i in range(1000):
        t0 = time.time()
        out_trt = trtModel(inputs)
        latency.append(time.time() - t0)
    print("Average trt Inference time = {} ms".format(
        format(sum(latency) * 1000 / len(latency), '.2f')))

    ## test trt fp16
    trtModel = TRTModel(tensorrt_path_fp16, (batch_size, 3, 48, 2000))
    out_trt = trtModel(inputs)[0]
    std = np.std(out_onnx - out_trt)
    print("std:", std)
    latency = []
    for i in range(1000):
        t0 = time.time()
        out_trt = trtModel(inputs)
        latency.append(time.time() - t0)
    print("Average trtfp16 Inference time = {} ms".format(
        format(sum(latency) * 1000 / len(latency), '.2f')))
