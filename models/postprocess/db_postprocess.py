"""
This code is refered from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/post_processing/seg_detector_representer.py
"""

import numpy as np
import cv2
import torch
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 adjust_ratio=0.01,
                 use_dilation=False,
                 score_mode="fast",
                 visual_output=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.adjust_ratio = adjust_ratio
        # self.process_box = process_box
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1],
                                                                       [1, 1]])
        self.visual = visual_output

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0,
                                dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0,
                                dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)

        # compute adjustment based on width and height of the box
        L = box[1][0]-box[0][0]
        H = box[3][1]-box[0][1]
        ratio = L/H
        # print(L,H ,area, ratio)
        adjust = ratio * self.adjust_ratio
        # print(adjust)
        distance = poly.area * (unclip_ratio+adjust) / poly.length
        # print("area:",poly.area, "length:",poly.length, "dis:",distance)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def visual_output(self, pred):
        im = np.array(pred[0] * 255).astype(np.uint8)
        cv2.imwrite("db_probability_map.png", im)
        print("The probalibity map is visualized in db_probability_map.png")

    def line_gradient_iterator(self, post_result, src_img):
        edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
        for i, bbox in enumerate(post_result[0]['points']):
            # bbox = BackgroundRemoval(bbox, src_img, i)
            boxW = bbox[0, 0] - bbox[1, 0]
            boxH = bbox[2, 1] - bbox[1, 1]
            for idx, e in enumerate(edges):
                p1 = (bbox[e[0], 0], bbox[e[0], 1])
                p2 = (bbox[e[1], 0], bbox[e[1], 1])
                p1_new, p2_new, check = self.checkLineVariation(
                    p1,
                    p2,
                    src_img,
                    idx,
                    max_expansion=0.3 * boxH,
                    threshold=200)
                if (check):
                    bbox[e[0]] = p1_new
                    bbox[e[1]] = p2_new
                else:
                    bbox[e[0]] = p1
                    bbox[e[1]] = p2

            post_result[0]['points'][i] = bbox
        return post_result

    def createLineIterator(self, P1, P2, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]
        #difference and absolute difference between points
        #used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)
        #predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        #Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  #vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  #horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  #diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                # slope = dX.astype(np.float32) / dY.astype(np.float32)
                slope = np.float32(dX) / np.float32(dY)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope *
                                  (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
            else:
                slope = np.float32(dY) / np.float32(dX)
                # slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope *
                                  (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

        #Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) &
                            (colY < imageH)]

        #Get intensities from img ndarray
        itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint),
                             itbuffer[:, 0].astype(np.uint)]

        return itbuffer[:, 2]

    def checkLineVariation(self,
                           P1,
                           P2,
                           src_img,
                           edge_idx,
                           threshold=30,
                           d_cumm=0,
                           max_expansion=20):
        d = 0.3 * max_expansion
        d = 8 if 0.3 * max_expansion > 8 else d
        pixels = self.createLineIterator(P1, P2, src_img)
        # print(pixels)
        grad = np.gradient(pixels)
        check = True

        if np.max(np.abs(grad)) > threshold:

            d_cumm += d

            if d_cumm >= max_expansion:
                return P1, P2, False

            if P2[0] == P1[0]:
                theta = np.pi / 2
            else:
                theta = np.arctan((P2[1] - P1[1]) / (P2[0] - P1[0]))
            dx = d * np.sin(theta)
            dy = d * np.cos(theta)

            if edge_idx == 0 or edge_idx == 1:
                P1 = (int(P1[0] - dx), int(P1[1] - dy))
                P2 = (int(P2[0] - dx), int(P2[1] - dy))
            if edge_idx == 2 or edge_idx == 3:
                P1 = (int(P1[0] + dx), int(P1[1] + dy))
                P2 = (int(P2[0] + dx), int(P2[1] + dy))

            P1, P2, check = self.checkLineVariation(
                P1,
                P2,
                src_img,
                edge_idx,
                threshold=threshold,
                d_cumm=d_cumm,
                max_expansion=max_expansion)

        return P1, P2, check

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().detach().numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        if self.visual:
            self.visual_output(pred)

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})

        # if self.process_box:
        #     src_img = src_img.squeeze(0).permute(1, 2, 0).numpy() * 255
        #     boxes_batch = self.line_gradient_iterator(boxes_batch, src_img)
        return boxes_batch


class DistillationDBPostProcess(object):

    def __init__(self,
                 model_name=["student"],
                 key=None,
                 thresh=0.3,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        self.model_name = model_name
        self.key = key
        self.post_process = DBPostProcess(thresh=thresh,
                                          box_thresh=box_thresh,
                                          max_candidates=max_candidates,
                                          unclip_ratio=unclip_ratio,
                                          use_dilation=use_dilation,
                                          score_mode=score_mode)

    def __call__(self, predicts, shape_list):
        results = {}
        for k in self.model_name:
            results[k] = self.post_process(predicts[k], shape_list=shape_list)
        return results
