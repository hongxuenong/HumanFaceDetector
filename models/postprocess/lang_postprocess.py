import torch
from ..dict.chars import CharsetMapper

class CharCorrectPost(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, max_text_length=25, **kwargs):
        self.charset = CharsetMapper(character_dict_path, max_text_length)

    def __call__(self, output, target, **kwargs):
        logits, pt_lengths = output['logits'], output['pt_lengths']
        text_output = []
        # for logit, pt_length in zip(logits, pt_lengths):
        #     text_output.append(self.charset.get_text(logit, pt_length))
        return output, target, text_output