import copy
import requests
import cv2
import numpy as np
from utility import parse_args, sorted_boxes, get_rotate_crop_image
from textDetector import TextDetector
from textRecognizer import TextRecognizer


def url_2_image(url):
    image_bytes = requests.get(url)
    image_np1 = np.frombuffer(image_bytes.content, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np1, cv2.IMREAD_COLOR)
    return image_np2


if __name__ == '__main__':

    # Read image
    img_hash = '3b8391d36f460a296e7c8ea85f8c166d'
    img_url = 'https://cf.shopee.sg/file/' + img_hash
    img = url_2_image(img_url)
    '''Model Initialization'''
    args = parse_args()
    args.det_model_dir = "models/det/dbnet_swinbase/det.onnx"  # det model weights
    args.rec_model_dir = "models/rec/rec_v3_mobile_dis_cn/cn.onnx"  # rec model weights
    args.rec_char_dict_path = "models/rec/rec_v3_mobile_dis_cn/cn_char.txt"  # char dict

    text_detector = TextDetector(args)
    text_recognizer = TextRecognizer(args)
    ori_im = img.copy()

    ## [Text detection] ##
    dt_boxes, elapse = text_detector(img)

    img_crop_list = []

    dt_boxes = sorted_boxes(dt_boxes)

    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)

    ## [Text Recognition] ##
    rec_res, elapse = text_recognizer(img_crop_list)

    # Filter
    filter_boxes, filter_rec_res = [], []
    for box, rec_result in zip(dt_boxes, rec_res):
        text, score = rec_result
        if score >= args.drop_score:
            filter_boxes.append(box)
            filter_rec_res.append(rec_result)

    res = [{
        "text": filter_rec_res[idx][0],
        "score": filter_rec_res[idx][1],
        "points": np.array(filter_boxes[idx]).astype(np.int32).tolist(),
    } for idx in range(len(filter_boxes))]

    print(res)