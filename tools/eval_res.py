import json
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.metrics.rec_metric import RecMetric


def run(metric, resFiles, gtFile, outputFile, student):
    res = {}
    gt = {}
    imgPath = {}

    if student:
        for resFile in resFiles:
            with open(resFile) as fr:
                for per in fr:
                    data = per.strip().split('\t')
                    imgName = data[0].split('/')[-1]
                    if len(data) == 2:
                        resJson = json.loads(data[1])
                        text = resJson['Student']['label']
                    else:
                        text = ""
                    res[imgName] = text
                    imgPath[imgName] = data[0]
    else:
        for resFile in resFiles:
            with open(resFile) as fr:
                for per in fr:
                    data = per.strip().split('\t')
                    imgName = data[0].split('/')[-1]
                    if len(data) >= 2:
                        text = data[1]
                    else:
                        text = ""
                    res[imgName] = text
                    imgPath[imgName] = data[0]

    with open(gtFile) as fr:
        for per in fr:
            data = per.strip().split('\t')
            imgName = data[0].split('/')[-1]

            if len(data) == 2:
                text = data[1]
            else:
                text = ""
            gt[imgName] = text
            
    resList = []
    gtList = []

    fw = open(outputFile, "w")
    for per in gt:
        if per in res:
            resList.append((res[per], 0))
            gtList.append((gt[per], 0))
            if metric._normalize_text(res[per]) != metric._normalize_text(
                    gt[per]):
                fw.write(imgPath[per] + "\t" + gt[per] + "\t" +
                         res[per] + "\n")
            # if "#" not in gt[per]:
            #     resList.append(converter.convert(res[per]))
            #     gtList.append(converter.convert(gt[per]))
    fw.close()
    res = metric((resList, gtList))
    res["num"] = len(resList)
    print(res)


if __name__ == '__main__':
    metric = RecMetric(lang='en')
    resFiles = [
        "output/rec/predicts_iter_en_1.txt"
    ]
    student = False
    gtFile = "data/EN/crops_eval_synth_correct_filter.txt"

    outputFile = "output/rec/en_res_ana_iter_1.txt"

    run(metric, resFiles, gtFile, outputFile, student)