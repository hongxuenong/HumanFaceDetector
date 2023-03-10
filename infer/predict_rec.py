import cv2
import numpy as np
import traceback
from textRecognizer import TextRecognizer
from utility import parse_args, get_image_file_list


def main():
    args = parse_args()

    image_file_list = get_image_file_list("data/CN/crops_eval/") # input image directory
    save_res_path = "output/rec/onnx_distillation_cn.txt" # save file path
    args.rec_model_dir = "models/rec/rec_v3_mobile_dis_cn/cn.onnx"  # model weights
    args.rec_char_dict_path = "models/rec/rec_v3_mobile_dis_cn/cn_char.txt"  # char dict of rec

    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []

    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, _ = text_recognizer(img_list)

    except Exception as E:
        print(traceback.format_exc())
        print(E)
        exit()

    with open(save_res_path, "w") as fout:
        for ino in range(len(img_list)):
            fout.write("{}\t{}\t{}\n".format(valid_image_file_list[ino],
                                             rec_res[ino][0], rec_res[ino][1]))


if __name__ == "__main__":
    main()
