import string
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from models.modeling.language.model_language import BCNLanguage

def _read_charset(filename):
    charset = []
    null_label = 0
    charset.append('<null>')
    with open(filename, 'rb') as f:
        for i, line in enumerate(f):
            # m = pattern.match(line)
            # assert m, f'Incorrect charset file. line #{i}: {line}'
            # label = int(m.group(1)) + 1
            # char = m.group(2)
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            charset.append(line)
    return charset


config={
  'Models':{
    'num_layers': 4,
    'loss_weight': 1.,
    'use_self_attn': False,
    'max_text_length': 39,
    'd_model': 512,
    'character_dict_path': 'ptocr/dict/en_char.txt'
  }

}

lm = BCNLanguage(config)
lm.load('output/lang_en_512/best_accuracy.ptparams')
lm = lm.eval()

# word = input('Target: ')
word = "MOMOCOVER"
word_gt = "MOMO COVER"

# gt


itos = _read_charset(config['Models']['character_dict_path']) # ['<null>'] + list(string.ascii_lowercase + '1234567890')
stoi = {s: i for i, s in enumerate(itos)}
max_len = config['Models']['max_text_length']

gt = [torch.as_tensor([stoi[c] for c in word_gt]), torch.arange(max_len + 1)]
gt = pad_sequence(gt, batch_first=True, padding_value=0)[:1]  # exclude dummy target
lengths_gt = torch.as_tensor([len(word_gt) + 1])
gt = F.one_hot(gt, len(itos)).float()

target = [torch.as_tensor([stoi[c] for c in word]), torch.arange(max_len + 1)]
target = pad_sequence(target, batch_first=True, padding_value=0)[:1]  # exclude dummy target
lengths = torch.as_tensor([len(word) + 1])

tgt = F.one_hot(target, len(itos)).float()

print(target, target.shape, lengths, lengths.shape)
res = lm(tgt, lengths)

pred = res['logits'].argmax(-1)
print(pred)
decoded = ''.join([itos[i] for i in pred.squeeze()])
decoded = decoded[:decoded.find('<null>')]
print(decoded)

# metirc
total_num_char = 0.
total_num_word = 0.
correct_num_char = 0.
correct_num_word = 0.

logits, pt_lengths = res['logits'], res['pt_lengths']
gt_labels, gt_lengths = gt,lengths_gt

for logit, pt_length, label, length in zip(logits, pt_lengths, gt_labels, gt_lengths):
    word_flag = True
    for i in range(length):
        char_logit = logit[i].topk(5)[1]
        char_label = label[i].argmax(-1)
        if char_label in char_logit: 
            correct_num_char += 1 
            print('1')
        else: 
            word_flag = False
            print('0')
        total_num_char += 1
    if pt_length == length and word_flag:
        correct_num_word += 1
    total_num_word += 1

mets = [correct_num_char / total_num_char,
        correct_num_word / total_num_word]
print(mets)