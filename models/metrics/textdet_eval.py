#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
# from triton.src.tritonRec import TritonRec
import cv2 as cv
"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""
# tritonP = TritonRec()


class DetectionIoUEvaluator(object):

    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, img, gt, pred):

        def getIOU(pG, pD):
            # img: [C, H, W]
            dmask = np.zeros((img.shape[-1], img.shape[-2]), dtype=np.int8)
            gmask = np.zeros((img.shape[-1], img.shape[-2]), dtype=np.int8)
            pD = np.array(pD)
            pG = np.array(pG)
            pD = np.array(np.round(pD), dtype=np.uint)
            pG = np.array(np.round(pG), dtype=np.uint)
            cv.fillPoly(dmask, pts=pD, color=1)
            cv.fillPoly(gmask, pts=pG, color=1)

            intersection = np.logical_and(dmask, gmask)
            union = np.logical_or(dmask, gmask)
            iou_score = np.sum(intersection) / np.sum(union)

            return np.sum(intersection) / np.sum(dmask), np.sum(
                intersection) / np.sum(gmask), iou_score

        # getIOU(gt, pred)

        perSampleMetrics = {}
        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        numGtCare = 0
        numDetCare = 0
        numDetCare_filtered = 0
        numGtCare_filtered = 0

        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            # if dontCare:
            #     gtDontCarePolsNum.append(len(gtPols) - 1)

        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue
            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)

        if len(gtPols) > 0 and len(detPols) > 0:
            # 1. Compute pair-wise IoU, intersactionD, interactionG
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            intersactionMatD = np.empty(outputShape)
            intersactionMatG = np.empty(outputShape)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    intersactionD, intersactionG, iou = getIOU([pG], [pD])
                    intersactionMatD[gtNum, detNum] = intersactionD
                    intersactionMatG[gtNum, detNum] = intersactionG
                    iouMat[gtNum, detNum] = iou

            # 2. Initialize pred/gt unpaired collections with all index
            unpaired_gt = np.where(np.sum(intersactionMatG, 1) < 0.3)[0]
            unpaired_pred = np.where(np.sum(intersactionMatD, 0) < 0.3)[0]

            # 3. 	2. Get evaluation pairs
            evaluation_pairs = []
            for gtNum in range(len(gtPols)):
                if gtNum in unpaired_gt:
                    continue
                paired_pred = np.where(intersactionMatD[gtNum, :] > 0.5)[0]

                if (len(paired_pred) > 0):
                    paireddetPols = [detPols[i] for i in paired_pred]
                    evaluation_pairs.append([[gtPols[gtNum]], paireddetPols])

            for detNum in range(len(detPols)):
                if detNum in unpaired_pred:
                    continue
                paired_gt = np.where(intersactionMatG[:, detNum] > 0.5)[0]
                if (len(paired_gt) > 0):
                    pairedgtPols = [gtPols[i] for i in paired_gt]
                    evaluation_pairs.append([pairedgtPols, [detPols[detNum]]])

            # num_pairs = len(evaluation_pairs)

            # 4. For each unpaired pred/gt, filter using recognition score
            #    a. If rec(pred) > 0.6 , append to evaluation pairs
            #    b. If rec(gt) < 0.3, remove from unpair gt
            # unpaired_pred_filter = list(unpaired_pred.copy())
            # unpaired_gt_filter = list(unpaired_gt.copy())

            # if (fix_gt):
            #     for detNum in unpaired_pred_filter:
            #         pred = detPols[detNum]
            #         rect = cv.minAreaRect(np.int0(pred))
            #         img_crop, img_rot = crop_rect(img, rect)
            #         img_flip = cv.flip(img_crop, 1)
            #         res = tritonP(img_crop, lang, img_crop.shape[0],
            #                       img_crop.shape[1])
            #         res_flip = tritonP(img_flip, lang, img_flip.shape[0],
            #                            img_flip.shape[1])
            #         res = res if res > res_flip else res_flip
            #         if (len(res)) > 0:
            #             if res['score'] > 0.6:
            #                 # evaluation_pairs.append([[pred], [pred]])
            #                 num_pairs += 1
            #                 unpaired_pred_filter.remove(detNum)

            #     for gtNum in unpaired_gt_filter:
            #         gt = gtPols[gtNum]
            #         rect = cv.minAreaRect(np.int0(gt))

            #         img_crop, img_rot = crop_rect(img, rect)
            #         img_flip = cv.flip(img_crop, 1)
            #         res = tritonP(img_crop, lang, img_crop.shape[0],
            #                       img_crop.shape[1])
            #         res_flip = tritonP(img_flip, lang, img_flip.shape[0],
            #                            img_flip.shape[1])
            #         res = res if res > res_flip else res_flip
            #         if (len(res)) > 0:
            #             if res['score'] < 0.3:
            #                 unpaired_gt_filter.remove(gtNum)

            # numGtCare_filtered = num_pairs + len(unpaired_gt)
            # numDetCare_filtered = num_pairs + len(unpaired_pred)

            scores = []
            for p in evaluation_pairs:
                scores.append(getIOU(p[0], p[1])[2])

            tp = np.sum(np.array(scores) > self.iou_constraint)
            precision = tp / (len(scores) + len(unpaired_pred))
            # print(precision)
            recall = tp / (len(scores) + len(unpaired_gt))
            # print(recall)
            hmean = 0 if (precision + recall) == 0 else 2.0 * \
                                                    precision * recall / (precision + recall)

            numGtCare = len(scores) + len(unpaired_gt)
            numDetCare = len(scores) + len(unpaired_pred)
            detMatched = tp

        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(detPols) > 100 else iouMat.tolist(),
            'gtPolPoints': gtPolPoints,
            'detPolPoints': detPolPoints,
            'gtCare': numGtCare,
            'detCare': numDetCare,
            # 'gtCare_filtered': numGtCare_filtered,
            # 'detCare_filtered': numDetCare_filtered,
            'gtDontCare': gtDontCarePolsNum,
            'detDontCare': detDontCarePolsNum,
            'detMatched': detMatched,
            'evaluationLog': evaluationLog
        }

        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)

        # for result in results:
        #     numGlobalCareGt += result['gtCare_filtered']
        #     numGlobalCareDet += result['detCare_filtered']
        #     matchedSum += result['detMatched']

        # methodRecall_filtered = 0 if numGlobalCareGt == 0 else float(
        #     matchedSum) / numGlobalCareGt
        # methodPrecision_filtered = 0 if numGlobalCareDet == 0 else float(
        #     matchedSum) / numGlobalCareDet
        # methodHmean_filtered = 0 if methodRecall + methodPrecision == 0 else 2 * \
        #                                                             methodRecall * methodPrecision / (
        #                                                                     methodRecall + methodPrecision)
        # sys.exit(-1)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
            # 'filtered_precision': methodPrecision_filtered,
            # 'filtered_recall': methodRecall_filtered,
            # 'filtered_hmean': methodHmean_filtered
        }

        return methodMetrics


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv.warpAffine(img, M, (width, height))
    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv.getRectSubPix(img_rot, size, center)

    if (img_crop.shape[0] > img_crop.shape[1]):
        img_crop = cv.rotate(img_crop, cv.ROTATE_90_CLOCKWISE)

    return img_crop, img_rot


if __name__ == '__main__':
    evaluator = DetectionIoUEvaluator()
    gts = [[{
        'points': [(0, 0), (20, 0), (20, 10), (0, 10)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(20, 20), (40, 20), (40, 40), (20, 40)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(20, 50), (40, 50), (55, 55), (55, 40)],
        'text': 1234,
        'ignore': False,
    }]]
    preds = [[{
        'points': [(0, 0), (10, 0), (10, 10), (0, 10)],
        'text': 123,
        'ignore': False,
    }, {
        'points': [(11.1, 0), (11.1, 11.2), (20, 11), (20, 0)],
        'text': 123,
        'ignore': False,
    }, {
        'points': [(20, 20), (55, 20), (55, 55), (20, 55)],
        'text': 123,
        'ignore': False,
    }, {
        'points': [(50, 50), (125, 50), (125, 125), (50, 125)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    # evaluator.evaluate_image(np.zeros((100, 100)), gts[0], preds[0])
    for gt, pred in zip(gts, preds):

        results.append(evaluator.evaluate_image(np.ones((200, 200)), gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)