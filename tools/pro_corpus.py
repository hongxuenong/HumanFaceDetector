import os
import sys
import random
import re
import string
from tqdm import tqdm
import pandas as pd

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.dict.chars import symFilter

def is_digit(text, ratio=0.5):
    length = max(len(text), 1)
    digit_num = sum([t in string.digits for t in text])
    if digit_num / length < ratio: return False
    return True


def disturb(word, degree, p=0.3):
    if len(word) // 2 < degree: return word
    if is_digit(word): return word
    if random.random() < p: return word
    else:
        index = list(range(len(word)))
        random.shuffle(index)
        index = index[:degree]
        new_word = []
        for i in range(len(word)):
            if i not in index: 
                new_word.append(word[i])
                continue
            if (word[i] not in string.ascii_letters) and (word[i] not in string.digits):
                # special token
                new_word.append(word[i])
                continue
            op = random.random()
            if op < 0.1: # add
                new_word.append(random.choice(string.ascii_letters))
                new_word.append(word[i])
            elif op < 0.2: continue  # remove
            else: new_word.append(random.choice(string.ascii_letters))  # replace
        return ''.join(new_word)


max_length = 40
min_length = 2
root = 'data'
lang = 'en'
# charset = 'abcdefghijklmnopqrstuvwxyz'
# digits = '0123456789'


with open('/data/OCR_data/langdata/tompi.en', 'r') as file:
    lines = file.readlines()


# generate training dataset
inp, gt = [], []
for line in tqdm(lines):
    # token = line.lower().split()
    text = ''.join(filter(lambda x: x in symFilter[lang], line.strip()))

    if len(text) < max_length:
        pass
    else:
        words = text.split()
        wlist = [word for word in words]
        index_num = random.randint(0,len(wlist)-1)
        text = wlist[index_num]
        if len(text) >= max_length: continue
        if index_num < len(wlist)-1: 
            for i in range(index_num+1, len(wlist)):
                if len(text) + len(wlist[i]) + 1 >= max_length:
                    break
                else:
                    text += (' '+wlist[i])
    if len(text) < min_length:
        continue
    inp.append(text)
    gt.append(text)

train_voc = os.path.join(root, 'ShopeeText-EN-train.csv')
pd.DataFrame({'inp':inp, 'gt':gt}).to_csv(train_voc, index=None, sep='\t')



# generate eval dataset
lines = inp
degree = 1
keep_num = 50000

random.shuffle(lines)
part_lines = lines[:keep_num]
inp, gt = [], []

for w in part_lines:
    new_w = disturb(w, degree)
    inp.append(new_w)
    gt.append(w)
    
eval_voc = os.path.join(root, f'ShopeeText-EN-eval_d{degree}.csv')
pd.DataFrame({'inp':inp, 'gt':gt}).to_csv(eval_voc, index=None, sep='\t')