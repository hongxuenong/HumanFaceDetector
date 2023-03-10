import numpy as np
import os

import sys
import cv2
import json
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.data import create_operators, transform
from models.modeling.architectures import build_model
from models.postprocess import build_post_process
from models.utils.save_load import load_model
from models.utils.utility import get_image_file_list, ArgsParser, read_pytorch_weights
from models.utils.config import load_config, merge_config


def draw_det_res(dt_boxes, config, img, img_name, save_path):
    if len(dt_boxes) > 0:
        import cv2
        src_im = img
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box],
                          True,
                          color=(255, 255, 0),
                          thickness=2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        print("The detected Image saved in {}".format(save_path))


def main(config, device):
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])

    # load_model(config, model)
    weights = read_pytorch_weights(global_config['pretrained_model'], device)
    model.load_state_dict(weights)

    # build post process
    post_process_class = build_post_process(config['PostProcess'])

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)

    ops = create_operators(transforms, global_config)

    save_res_path = config['Global']['save_res_path']
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.to(device).eval()

    with open(save_res_path, "wb") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            print("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.Tensor(images)
            with torch.no_grad():
                preds = model(images.to(device))
            post_result = post_process_class(preds, shape_list)

            src_img = cv2.imread(file)

            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json['points'] = box.tolist()
                        dt_boxes_list.append(tmp_json)
                    dt_boxes_json = dt_boxes_list
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(
                        config['Global']
                        ['save_res_path']) + "/det_results_{}/".format(k)
                    draw_det_res(boxes, config, src_img, file, save_det_path)
            else:
                boxes = post_result[0]['points']
                dt_boxes_json = []
                # write result
                for box in boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json['points'] = box.tolist()
                    dt_boxes_json.append(tmp_json)
                save_det_path = os.path.dirname(
                    config['Global']['save_res_path']) + "/det_results/"
                draw_det_res(boxes, config, src_img, file, save_det_path)
            otstr = file + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())

    print("success!")


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    device = torch.device("cuda" if config['Global']['use_gpu'] else "cpu")
    main(config, device)

## python tools/infer_det.py -c configs/det/det_swin_db_base.yml -o Global.pretrained_model=models/det/dbnet_swinbase/mmocr_pretrained_dbnet_swinbase_best.pth Global.infer_img=data/Det_Multi/imgs_eval Global.save_res_path="output/det/det_swin_db_base.txt"
