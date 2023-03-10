import json
import copy
import os
import cv2
import numpy as np
from utility import parse_args, sorted_boxes, get_rotate_crop_image, get_image_file_list
from textDetector import TextDetector
from textRecognizer import TextRecognizer


def main():
    save_res_path = "output/e2e/cn.txt" # save file path
    image_file_list = get_image_file_list("data/Det_Multi/imgs_eval") # input image directory

    # Model Initialization
    args = parse_args()
    args.det_model_dir = "models/det/dbnet_swinbase/det.onnx"  # det model weights
    args.rec_model_dir = "models/rec/rec_v3_mobile_dis_cn/cn.onnx"  # rec model weights
    args.rec_char_dict_path = "models/rec/rec_v3_mobile_dis_cn/cn_char.txt"  # char dict of rec
    text_detector = TextDetector(args)
    text_recognizer = TextRecognizer(args)

    fw = open(save_res_path, 'w')
    for image_file in image_file_list:
        img = cv2.imread(image_file)

        ## [Text detection] ##
        dt_boxes, _ = text_detector(img)

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)

        ## [Text Recognition] ##
        rec_res, _ = text_recognizer(img_crop_list)

        # Filter
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= 0:  # args.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        res = [{
            "text": filter_rec_res[idx][0],
            "score": filter_rec_res[idx][1],
            "points": np.array(filter_boxes[idx]).astype(np.int32).tolist(),
        } for idx in range(len(filter_boxes))]

        print(res)
        fw.write(image_file + "\t" + json.dumps(res, ensure_ascii=False) +
                 '\n')

    fw.close()


if __name__ == "__main__":
    main()