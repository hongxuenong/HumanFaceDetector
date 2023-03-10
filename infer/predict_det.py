import os
import time
import json
import numpy as np
import cv2
from textDetector import TextDetector
from utility import parse_args, get_image_file_list, draw_text_det_res


def main():
    args = parse_args()

    image_file_list = get_image_file_list("data/Det_Multi/imgs_eval") # input image directory
    draw_img_save = "output/det/predict" # save directory
    args.det_model_dir = "models/det/dbnet_swinbase/det.onnx"  # det model weights

    text_detector = TextDetector(args)
    count = 0
    total_time = 0

    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = text_detector(img)

    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)

    fw = open(os.path.join(draw_img_save, "det_results.txt"), 'w')
    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        st = time.time()
        dt_boxes, _ = text_detector(img)
        elapse = time.time() - st
        if count > 0:
            total_time += elapse
        count += 1
        save_pred = image_file + "\t" + str(
            json.dumps([x.tolist() for x in dt_boxes])) + "\n"
        fw.write(save_pred + '\n')
        print(save_pred)
        print("The predict time of {}: {}".format(image_file, elapse))
        src_im = draw_text_det_res(dt_boxes, image_file)
        img_name_pure = os.path.split(image_file)[-1]
        img_path = os.path.join(draw_img_save,
                                "det_res_{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
        print("The visualized image saved in {}".format(img_path))
        total_time += elapse
    print(total_time)

    fw.close()


if __name__ == "__main__":
    main()
