import torch
import numpy as np
import os
import json
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.data.imaug import create_operators, transform
from models.modeling.architectures import build_model
from models.postprocess import build_post_process
from models.utils.utility import get_image_file_list, read_pytorch_weights, ArgsParser
from models.utils.config import load_config, merge_config, add_config_rec


def main(config, device):
    global_config = config['Global']

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config = add_config_rec(config, char_num)

    model = build_model(config['Architecture'])

    # load_model(config, model)
    weights = read_pytorch_weights(global_config['pretrained_model'], device)
    model.load_state_dict(weights)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config['Global'].get('save_res_path',
                                         "./output/rec/predicts.txt")
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.to(device).eval()

    with open(save_res_path, "w") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            print("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            if config['Architecture']['algorithm'] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                others = [
                    torch.tensor(encoder_word_pos_list).to(device),
                    torch.tensor(gsrm_word_pos_list).to(device),
                    torch.tensor(gsrm_slf_attn_bias1_list).to(device),
                    torch.tensor(gsrm_slf_attn_bias2_list).to(device)
                ]

            if config['Architecture']['algorithm'] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                img_metas = [torch.tensor(valid_ratio).to(device)]

            images = np.expand_dims(batch[0], axis=0)
            images = torch.tensor(images).to(device)

            with torch.no_grad():
                if config['Architecture']['algorithm'] == "SRN":
                    preds = model(images, others)
                elif config['Architecture']['algorithm'] == "SAR":
                    preds = model(images, img_metas)
                elif config['Architecture']['algorithm'] == "ABINetIter":
                    preds = model(images, post_process_class)
                else:
                    preds = model(images)

            post_result = post_process_class(preds)
            info = None
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info, ensure_ascii=False)
            else:
                if len(post_result[0]) >= 2:
                    info = post_result[0][0] + "\t" + str(post_result[0][1])

            if info is not None:
                print("\t result: {}".format(info))
                fout.write(file + "\t" + info + "\n")
    print("success!")


if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    device = torch.device("cuda" if config['Global']['use_gpu'] else "cpu")
    main(config, device)

# python tools/infer_rec.py -c convert/rec_v3_model_convert.yml -o Global.pretrained_model=output/convert/rec_cn_v3.pth  Global.infer_img=/ldap_home/jingze.chi/PaddleOCR/data/CN_SG_5K/crops_eval_synth/ Global.character_dict_path=/ldap_home/jingze.chi/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt Global.save_res_path=./output/rec/predicts_cn.txt
