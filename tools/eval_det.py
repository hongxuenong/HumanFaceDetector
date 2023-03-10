import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import os
from models.modeling.architectures import build_model
import tools.program as program
from models.metrics.textdet_eval import DetectionIoUEvaluator
from tqdm import tqdm
import cv2 as cv
import numpy as np
import json
from models.utils.utility import read_pytorch_weights
from models.utils.save_load import load_pretrained_params
from models.data import create_operators, transform
from models.postprocess import build_post_process


def visualize(img, bboxes, savepath, color=[255, 0, 0], thickness=2):
    for box in bboxes:
        bbox = box['points']
        bbox = np.array(bbox).astype(np.int32)
        cv.polylines(img, [bbox[:8].reshape(-1, 2)],
                     True,
                     color=color,
                     thickness=thickness)
        if ('score' in box):
            img = cv.putText(img,
                             str(box['score'])[:5],
                             (int(bbox[0][0]), int(bbox[0][1])),
                             cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                             cv.LINE_AA)
    cv.imwrite(savepath, img)
    return img


if __name__ == '__main__':
    config, device, logger, writer = program.preprocess(is_train=False)
    model = build_model(config['Architecture'])
    global_config = config['Global']
    # weights = read_pytorch_weights(global_config['pretrained_model'], device)
    # model.load_state_dict(weights)
    pretrained_model = global_config.get('pretrained_model')
    if pretrained_model:
        load_pretrained_params(model, pretrained_model, logger)

    evaluator = DetectionIoUEvaluator(iou_constraint=0.5)

    imgDir = config['Eval']['dataset']['data_dir']
    gtFile = config['Eval']['dataset']['label_file_list']

    with open(gtFile) as fr:
        gtlines = fr.readlines()

    print("number of files:", len(gtlines))

    preds = []
    gts = []

    results = []

    scale = 1
    total_time = 0
    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)
    global_config = config['Global']
    ops = create_operators(transforms, global_config)
    post_process_class = build_post_process(config['PostProcess'])
    for line in tqdm(gtlines):
        try:
            imagePath = line.split('\t')[0]
            gt_info = json.loads(line.split('\t')[1])
            # print(imgDir + imagePath)
            img = cv.imread(imgDir + imagePath)
            with open(imgDir + imagePath, 'rb') as f:
                image = f.read()
                data = {'image': image}
            batch = transform(data, ops)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = torch.Tensor(images)
            outs = model(images)
            post_result = post_process_class(outs, shape_list)
        except:
            continue

        pred = []
        boxes = post_result['Student'][0]['points']
        for k, box in enumerate(boxes):
            points = box
            rect = cv.minAreaRect(np.int0(points))

            points = cv.boxPoints(rect)
            points = [(int(np.round(p[0])), int(np.round(p[1])))
                      for p in points]
            pred.append({
                'points': points,
                'ignore': False,
            })

        gt = []
        gtlines = []

        for b in gt_info:
            gt.append({
                'points': np.array(b['points']),
                'ignore': False,
            })

        if (len(gt) == 0):
            #skip if no gt box
            continue
        gts.append(gt)
        preds.append(pred)

        # get evaluation result
        test = evaluator.evaluate_image(img.transpose(2, 0, 1), gt, pred)
        results.append(test)
        precision, recall, hmean = test['precision'], test['recall'], test[
            'hmean']

        # # print(precision, recall, hmean)
        print(imagePath)
        img_savename = 'test/' + imagePath.split('/')[-1].split(
            '.'
        )[0] + '_p' + f'{precision:.2f}' + '_r' + f'{recall:.2f}' + '_h' + f'{hmean:.2f}' + '.png'

        visualize(img, pred, img_savename)

    metrics = evaluator.combine_results(results)
    print(metrics)

# /ldap_home/xuenong.hong/anaconda3/envs/open-mmlab/bin/python /ldap_home/xuenong.hong/txthdl/projects/tritonocr/tools/eval_det.py -c configs/det/det_swin_db_dml.yml