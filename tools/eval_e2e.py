from collections import namedtuple
import json
import os
import sys
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
from shapely.geometry import Polygon
"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.metrics.rec_metric import RecMetric


class E2EEvaluator(object):

    def __init__(self,
                 lang,
                 iou_constraint=0.5,
                 area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint
        self.langMet = RecMetric(lang=lang)

    def evaluate_image(self, gt, pred):

        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        gtTxts = []
        detTxts = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ""

        # print(len(gt))
        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            if 'ignore' not in gt[n]:
                dontCare = False
            else:
                dontCare = gt[n]['ignore']
            #             points = Polygon(points)
            #             points = points.buffer(0)
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            gtTxts.append(gt[n]['text'])
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) +
            " don't care)\n" if len(gtDontCarePolsNum) > 0 else "\n")

        for n in range(len(pred)):
            points = pred[n]['points']
            #             points = Polygon(points)
            #             points = points.buffer(0)
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            detTxts.append(pred[n]['text'])
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += "DET polygons: " + str(len(detPols)) + (
            " (" + str(len(detDontCarePolsNum)) +
            " don't care)\n" if len(detDontCarePolsNum) > 0 else "\n")

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        # if iouMat[gtNum, detNum] > self.iou_constraint:
                        if iouMat[
                                gtNum,
                                detNum] > self.iou_constraint and self.langMet._normalize_text(
                                    gtTxts[gtNum]
                                ) == self.langMet._normalize_text(
                                    detTxts[detNum]):
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(
                detMatched) / numDetCare

        hmean = 0 if (precision + recall) == 0 else 2.0 * \
                                                    precision * recall / (precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'detMatched': detMatched,
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
        # print(methodRecall, methodPrecision, methodHmean)
        # sys.exit(-1)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        return methodMetrics


def draw(image, imgHash, res, saveDir):
    fontPIL = ImageFont.truetype('./data/font/simfang.ttf', 20)
    draw = ImageDraw.Draw(image)
    for i, resLine in enumerate(res):
        text = resLine["text"]
        box = resLine["points"]
        draw.text(box[0], text, font=fontPIL, fill=(0, 0, 0), anchor="ld")
        draw.polygon([
            box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1],
            box[3][0], box[3][1]
        ],
                     outline=(0, 255, 0),
                     width=2)

    src = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    dst = saveDir + imgHash + ".jpg"
    cv2.imwrite(dst, src)


def run():
    lang = "CN"
    dataDir = "data/e2e/" + lang
    evaluator = E2EEvaluator(lang.lower())
    visualization = True

    gtPath = dataDir + "/gt.txt"
    imgDir = dataDir + "/ori/"
    predDirTencent = dataDir + "/tencent/"
    predDirBaidu = dataDir + "/baidu/"
    resFile = dataDir + "/res.txt"
    saveTencent = dataDir + "/tencent_vis/"
    saveBaidu = dataDir + "/baidu_vis/"
    saveRes = dataDir + "/res_vis/"

    # ground truth
    gtDict = {}
    with open(gtPath) as fr:
        for line in fr:
            imgName, data = line.strip().split("\t")
            data = json.loads(data)
            gtList = []
            for perData in data:
                gtList.append({
                    "points": perData["points"],
                    'text': perData["transcription"],
                    'ignore': (perData["transcription"] == "###")
                })
            gtDict[imgName.replace(".jpg", "")] = gtList

    # tencent
    results = []
    for imgHash in gtDict:
        try:
            with open(predDirTencent + imgHash + ".json", 'r') as f:
                # with open(predDirTencent + imgHash, 'r') as f:
                predJson = json.load(f)
        except:
            continue
        predList = []
        for res in predJson["TextDetections"]:
            resPoint = []
            for p in res["Polygon"]:
                resPoint.append((p["X"], p["Y"]))
            predList.append({"points": resPoint, 'text': res["DetectedText"]})
        results.append(evaluator.evaluate_image(gtDict[imgHash], predList))
        if visualization:
            image = Image.open(imgDir + imgHash + ".jpg")
            draw(image, imgHash, predList, saveTencent)
    metrics = evaluator.combine_results(results)
    print("tencent", metrics)

    # baidu
    results = []
    for imgHash in gtDict:
        try:
            with open(predDirBaidu + imgHash + ".json", 'r') as f:
                # with open(predDirBaidu + imgHash, 'r') as f:
                predJson = json.load(f)
        except:
            continue
        predList = []
        for res in predJson["words_result"]:
            resPoint = [(res["location"]["left"], res["location"]["top"]),
                        (res["location"]["left"] + res["location"]["width"],
                         res["location"]["top"]),
                        (res["location"]["left"] + res["location"]["width"],
                         res["location"]["top"] + res["location"]["height"]),
                        (res["location"]["left"],
                         res["location"]["top"] + res["location"]["height"])]
            predList.append({"points": resPoint, 'text': res["words"]})
        results.append(evaluator.evaluate_image(gtDict[imgHash], predList))
        if visualization:
            image = Image.open(imgDir + imgHash + ".jpg")
            draw(image, imgHash, predList, saveBaidu)
    metrics = evaluator.combine_results(results)
    print("baidu", metrics)

    # triton
    results = []
    tritonRes = {}
    with open(resFile) as fr:
        for line in fr:
            imgName, data = line.strip().split("\t")
            data = json.loads(data)
            predList = []
            for perData in data:
                predList.append({
                    "points": perData["points"],
                    'text': perData["text"]
                })
            tritonRes[os.path.basename(imgName).replace(".jpg", "")] = predList
    for imgHash in gtDict:
        results.append(
            evaluator.evaluate_image(gtDict[imgHash], tritonRes[imgHash]))
        if visualization:
            image = Image.open(imgDir + imgHash + ".jpg")
            draw(image, imgHash, tritonRes[imgHash], saveRes)
    metrics = evaluator.combine_results(results)
    print("triton", metrics)


if __name__ == '__main__':
    run()
    # evaluator = E2EEvaluator()
    # gts = [[{
    #     'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
    #     'text': 123,
    #     'ignore': False,
    # }, {
    #     'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
    #     'text': 5678,
    #     'ignore': False,
    # }]]
    # preds = [[{
    #     'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
    #     'text': 123,
    #     'ignore': False,
    # }]]
    # results = []
    # for gt, pred in zip(gts, preds):
    #     results.append(evaluator.evaluate_image(gt, pred))
    # metrics = evaluator.combine_results(results)
    # print(metrics)
