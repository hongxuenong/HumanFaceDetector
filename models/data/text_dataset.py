import torch
from torch.utils.data import Dataset
import pandas as pd
from ..dict.chars import CharsetMapper


class TextDataSet(Dataset):

    def __init__(self, config, mode, logger, seed=None):
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        is_training = True if mode == 'Train' else False
        self.path = dataset_config['label_file']
        self.max_length = dataset_config['max_length']
        self.case_sensitive, self.use_sm = global_config[
            'case_sensitive'], dataset_config['use_sm']
        self.smooth_factor, self.smooth_label = dataset_config[
            'smooth_factor'], dataset_config['smooth_label']
        self.charset = CharsetMapper(global_config['character_dict_path'],
                                     max_length=self.max_length + 1)
        self.one_hot_x, self.one_hot_y, self.is_training = dataset_config[
            'one_hot_x'], dataset_config['one_hot_y'], is_training
        # if self.is_training and self.use_sm: self.sm = SpellingMutation(charset=self.charset)
        self.delimiter = dataset_config.get('delimiter', '\t')
        dtype = {'inp': str, 'gt': str}
        self.df = pd.read_csv(self.path,
                              dtype=dtype,
                              delimiter=self.delimiter,
                              na_filter=False)
        self.inp_col, self.gt_col = 0, 1
        self.need_reset = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_x = self.df.iloc[idx, self.inp_col]
        text_x = text_x[:self.max_length]

        if self.is_training and self.use_sm: text_x = self.sm(text_x)

        length_x = torch.tensor(len(text_x) + 1).to(
            dtype=torch.long)  # one for end token
        label_x = self.charset.get_labels(text_x, case_sensitive=self.case_sensitive)

        label_x = torch.tensor(label_x)
        if self.one_hot_x:
            label_x = onehot(label_x, self.charset.num_classes)
            if self.is_training and self.smooth_label:
                label_x = torch.stack(
                    [self.prob_smooth_label(l) for l in label_x])
        x = [label_x, length_x]

        text_y = self.df.iloc[idx, self.gt_col]
        text_y = text_y[:self.max_length]

        length_y = torch.tensor(len(text_y) + 1).to(
            dtype=torch.long)  # one for end token
        label_y = self.charset.get_labels(text_y,
                                          case_sensitive=self.case_sensitive)
        label_y = torch.tensor(label_y)
        if self.one_hot_y: label_y = onehot(label_y, self.charset.num_classes)
        y = [label_y, length_y]

        return x + y

    def prob_smooth_label(self, one_hot):
        one_hot = one_hot.float()
        delta = torch.rand([]) * self.smooth_factor
        num_classes = len(one_hot)
        noise = torch.rand(num_classes)
        noise = noise / noise.sum() * delta
        one_hot = one_hot * (1 - delta) + noise
        return one_hot


def onehot(label, depth, device=None):
    """ 
    Args:
        label: shape (n1, n2, ..., )
        depth: a scalar

    Returns:
        onehot: (n1, n2, ..., depth)
    """
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label, device=device)
    onehot = torch.zeros(label.size() + torch.Size([depth]), device=device)
    onehot = onehot.scatter_(-1, label.unsqueeze(-1), 1)

    return onehot

