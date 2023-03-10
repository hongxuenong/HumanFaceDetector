
import os
import sys
import numpy as np
import skimage
import torch
import signal
import random

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy
from torch.utils.data import Dataset, DataLoader, BatchSampler#, DistributedBatchSampler

from .imaug import transform, create_operators
from .simple_dataset import SimpleDataSet
from .wider_face import WiderFaceDetection
# from ppocr.data.lmdb_dataset import LMDBDataSet
# from ppocr.data.pgnet_dataset import PGDataSet
# from ppocr.data.pubtab_dataset import PubTabDataSet

__all__ = ['build_dataloader', 'transform', 'create_operators']


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


def build_dataloader(config, mode, logger, seed=None):
    config = copy.deepcopy(config)

    support_dict = [
        'SimpleDataSet', 'LMDBDataSet', 'PGDataSet', 'PubTabDataSet', 'TextDataSet','WiderFaceDetection'
    ]
    module_name = config[mode]['dataset']['name']
    assert module_name in support_dict, Exception(
        'DataSet only support {}'.format(support_dict))
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    dataset = eval(module_name)(config, mode, logger, seed)
    loader_config = config[mode]['loader']
    batch_size = loader_config['batch_size_per_card']
    drop_last = loader_config['drop_last']
    shuffle = loader_config['shuffle']
    num_workers = loader_config['num_workers']
    # if 'use_shared_memory' in loader_config.keys():
    #     use_shared_memory = loader_config['use_shared_memory']
    # else:
    #     use_shared_memory = True

    # if mode == "Train":
    #     # Distribute data to multiple cards
    #     batch_sampler = DistributedBatchSampler(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         drop_last=drop_last)
        
    # else:
    #     # Distribute data to single card
    #     batch_sampler = BatchSampler(
    #         dataset=dataset,
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #         drop_last=drop_last)

    if 'collate_fn' in loader_config:
        from . import collate_fn
        collate_fn = getattr(collate_fn, loader_config['collate_fn'])()
    else:
        collate_fn = None
    # data_loader = DataLoader(
    #     dataset=dataset,
    #     batch_sampler=batch_sampler,
    #     places=device,
    #     num_workers=num_workers,
    #     return_list=True,
    #     use_shared_memory=use_shared_memory,
    #     collate_fn=collate_fn)
    if mode == "Train":
        if config['Global']['distributed']:
            train_sampler = torch.utils.data.DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
        data_loader = DataLoader(
            dataset=dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn)
    else:
        data_loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn)
    
    # support exit using ctrl+c
    signal.signal(signal.SIGINT, term_mp)
    signal.signal(signal.SIGTERM, term_mp)

    return data_loader
