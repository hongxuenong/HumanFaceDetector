import sys
import os

import torch
from torch import nn
import torch.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.data import build_dataloader
from models.modeling.architectures import build_model
from models.losses import build_loss
from models.optimizer import build_optimizer
from models.postprocess import build_post_process
from models.metrics import build_metric
from models.utils.save_load import load_model, load_pretrained_params
from models.utils.utility import set_seed
from models.utils.config import add_config_rec
import tools.program as program
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'


def main(config, device, logger, vdl_writer):
    global_config = config['Global']

    ## init dist environment
    if config['Global']['distributed']:
        dist.init_process_group(backend="nccl")

    ## build dataloader
    train_dataloader = build_dataloader(config, 'Train', logger)
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n" +
            "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
            +
            "\t2. The annotation file and path in the configuration file are provided normally."
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', logger)
    else:
        valid_dataloader = None

    ## build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    ## build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config = add_config_rec(config, char_num)

    model = build_model(config['Architecture']).to(device)

    if config['Global']['distributed']:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(local_rank, "+++++++++++++++++++++++++")
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank)

    ## build loss
    loss_class = build_loss(config['Loss']).to(device)

    ## build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        model=model,
        logger=logger)

    ## build metric
    eval_class = build_metric(config['Metric'])

    ## load pretrain model
    pre_best_model_dict = load_model(config, model, device, logger, optimizer)

    logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))

    use_amp = config["Global"].get("use_amp", False)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    ## start train
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer, scaler)


def eval_model(config, device, logger):
    global_config = config['Global']

    ## build dataloader
    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', logger)
    else:
        valid_dataloader = None

    ## build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    ## build model
    # for rec algorithm
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        config = add_config_rec(config, char_num)

    model = build_model(config['Architecture']).to(device)

    ## build metric
    eval_class = build_metric(config['Metric'])

    ## load pretrained weights
    pretrained_model = global_config.get('pretrained_model')
    if pretrained_model:
        load_pretrained_params(model, pretrained_model, logger)

    if valid_dataloader is not None:
        logger.info('valid dataloader has {} iters'.format(
            len(valid_dataloader)))

    ## start eval
    cur_metric = program.eval(model,
                              device,
                              valid_dataloader,
                              post_process_class,
                              eval_class,
                              model_type='rec',
                              extra_input='SVTR')
    logger.info(cur_metric)


def test_reader(config, device, logger):
    loader = build_dataloader(config, 'Train', logger)
    import time
    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(
                    count, len(data[0]), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    is_train = True
    config, device, logger, writer = program.preprocess(is_train=is_train)

    # reduce randomness
    seed = config['Global']['seed'] if 'seed' in config['Global'] else 1024
    set_seed(seed)

    if config['Global']['distributed']:
        device = torch.device('cuda', int(os.environ["LOCAL_RANK"]))
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

    # test dataloader speed
    # test_reader(config, device, logger)

    if is_train:
        main(config, device, logger, writer)
    else:
        eval_model(config, device, logger)

# python tools/train.py -c configs/rec_v3_th.yml -o Global.pretrained_model=output/rec_v3_th/best_accuracy.ptparams
# torchrun --standalone --nnodes=1 --nproc_per_node=3 tools/train.py -c configs/rec_v3_distillation_th.yml