from ..dict.chars import CharsetMapper

class TopKTextAcc(object):
    _names = ['ccr', 'cwr']
    def __init__(self, k, charset_path, max_length, case_sensitive, main_indicator):
        self.k = k
        self.charset_path = charset_path
        self.main_indicator = main_indicator
        self.max_length = max_length
        self.case_sensitive = case_sensitive
        self.charset = CharsetMapper(charset_path, self.max_length)
        self.reset()

    def reset(self):
        self.total_num_char = 0.
        self.total_num_word = 0.
        self.correct_num_char = 0.
        self.correct_num_word = 0.

    def __call__(self, last_output, last_target, **kwargs):
        logits, pt_lengths = last_output['logits'], last_output['pt_lengths']
        gt_labels, gt_lengths = last_target[:]

        total_num_char = 0.
        total_num_word = 0.
        correct_num_char = 0.
        correct_num_word = 0.
        for logit, pt_length, label, length in zip(logits, pt_lengths, gt_labels, gt_lengths):
            word_flag = True
            for i in range(length):
                char_logit = logit[i].topk(self.k)[1]
                char_label = label[i].argmax(-1)
                if char_label in char_logit: correct_num_char += 1
                else: word_flag = False
                total_num_char += 1
            if pt_length == length and word_flag:
                correct_num_word += 1
            total_num_word += 1
        self.total_num_char += total_num_char
        self.total_num_word += total_num_word
        self.correct_num_char += correct_num_char
        self.correct_num_word += correct_num_word
        return {
            'ccr': self.correct_num_char / self.total_num_char,
            'cwr': self.correct_num_word / self.total_num_word
        }

    def get_metric(self):
        """
        return metrics {
                 'ccr': 0,
                 'cwr': 0,
            }
        """
        ccr = self.correct_num_char / self.total_num_char
        cwr = self.correct_num_word / self.total_num_word
        self.reset()
        return {'ccr': ccr, 'cwr': cwr}
