import os
import sys
import platform
import yaml
import time
import datetime
import torch
# from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
from argparse import ArgumentParser, RawDescriptionHelpFormatter

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
from models.utils.stats import TrainingStats, log_metrics
from models.utils.save_load import save_model
from models.utils.utility import print_dict, AverageMeter
from models.utils.logging import get_logger
from models.data import build_dataloader


class ArgsParser(ArgumentParser):

    def __init__(self):
        super(ArgsParser,
              self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument("-o",
                          "--opt",
                          nargs='+',
                          help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format ' \
                 '\"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def check_device(use_gpu):
    """
    Log error and exit when set use_gpu=true in torch
    cpu version.
    """
    err = "Config {} cannot be set as true while your torch " \
          "is not compiled with {} ! \nPlease try: \n" \
          "\t1. Install torch to run model on {} \n" \
          "\t2. Set {} as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not torch.cuda.is_available():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
    except Exception as e:
        pass


def train(config,
          train_dataloader,
          valid_dataloader,
          device,
          model,
          loss_class,
          optimizer,
          lr_scheduler,
          post_process_class,
          eval_class,
          pre_best_model_dict,
          logger,
          log_writer=None,
          scaler=None):

    ## load configs
    cal_metric_during_train = config['Global'].get('cal_metric_during_train', False)
    calc_epoch_interval = config['Global'].get('calc_epoch_interval', 1)
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']

    global_step = 0
    if 'global_step' in pre_best_model_dict:
        global_step = pre_best_model_dict['global_step']
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        if len(valid_dataloader) == 0:
            logger.info(
                'No Images in eval dataset, evaluation during training ' \
                'will be disabled'
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, " \
            "an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step))
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ['lr'])

    model.train()


    extra_input = False


    start_epoch = best_model_dict[
        'start_epoch'] if 'start_epoch' in best_model_dict else 1

    total_samples = 0
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    reader_start = time.time()
    eta_meter = AverageMeter()

    max_iter = len(train_dataloader) - 1 if platform.system(
    ) == "Windows" else len(train_dataloader)

    ## torch profiler 
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            config['Global']['save_model_dir']+'/log'),
        with_stack=True)
    profiler.start()

    ## start training
    for epoch in range(start_epoch, epoch_num + 1):
        if train_dataloader.dataset.need_reset:
            train_dataloader = build_dataloader(config, 'Train', logger, seed=epoch)
            max_iter = len(train_dataloader) - 1 if platform.system(
            ) == "Windows" else len(train_dataloader)

        for idx, batch in enumerate(train_dataloader):
            batch = [perData.to(device) for perData in batch]
            train_reader_cost += time.time() - reader_start
            if idx >= max_iter:
                break
            lr = optimizer.param_groups[0]['lr']
            images = batch[0]
            optimizer.zero_grad()

            # use amp
            if scaler:
                with autocast():
                    if extra_input:
                        preds = model(images, data=batch[1:])
                    else:
                        preds = model(images)
                    loss = loss_class(preds, batch)
            else:
                if extra_input: # for some algos need extra inputs
                    preds = model(images, data=batch[1:])
                elif model_type == 'lang': # language model
                    preds = model(batch[0], batch[1])
                else:
                    preds = model(images)

                loss = loss_class(preds, batch)

            avg_loss = loss['loss']

            # use amp
            if scaler:
                scaler.scale(avg_loss).backward(
                    torch.ones_like(avg_loss).to(device))
                scaler.step(optimizer)
                scaler.update()
            else:
                avg_loss.backward(torch.ones_like(avg_loss).to(device))
                optimizer.step()

            if epoch < 2:
                profiler.step()

            if cal_metric_during_train and epoch % calc_epoch_interval == 0:  # only rec and cls need
                batch = [item.cpu().numpy() for item in batch]
                if model_type == 'lang':
                    post_result = post_process_class(preds,
                                                     [batch[2], batch[3]])
                    eval_class(post_result[0], post_result[1])
                else:
                    if config['Loss']['name'] in ['MultiLoss', 'MultiLoss_v2'
                                                  ]:  # for multi head loss
                        post_result = post_process_class(
                            preds['ctc'], batch[1])  # for CTC head out
                    else:
                        post_result = post_process_class(preds, batch[1])
                    eval_class(post_result, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)

            train_batch_time = time.time() - reader_start
            train_batch_cost += train_batch_time
            eta_meter.update(train_batch_time)
            global_step += 1
            total_samples += len(images)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            ## logger and writer
            stats = {
                k: v.detach().cpu().numpy().mean()
                for k, v in loss.items()
            }
            stats['lr'] = lr
            train_stats.update(stats)

            if config['Global']['distributed']:
                local_rank = int(os.environ["LOCAL_RANK"])
            if log_writer is not None and (config['Global']['distributed']
                                           == False or local_rank == 0):
                log_metrics(log_writer,
                            metrics=train_stats.get(),
                            prefix="TRAIN",
                            step=global_step)

            if (config['Global']['distributed'] == False or local_rank
                    == 0) and ((global_step > 0
                                and global_step % print_batch_step == 0) or
                               (idx >= len(train_dataloader) - 1)):
                logs = train_stats.log()

                eta_sec = ((epoch_num + 1 - epoch) * \
                    len(train_dataloader) - idx - 1) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                strs = 'epoch: [{}/{}], global_step: {}, {}, avg_reader_cost: ' \
                       '{:.5f} s, avg_batch_cost: {:.5f} s, avg_samples: {}, ' \
                       'ips: {:.5f} samples/s, eta: {}'.format(
                    epoch, epoch_num, global_step, logs,
                    train_reader_cost / print_batch_step,
                    train_batch_cost / print_batch_step,
                    total_samples / print_batch_step,
                    total_samples / train_batch_cost, eta_sec_format)
                logger.info(strs)

                total_samples = 0
                train_reader_cost = 0.0
                train_batch_cost = 0.0

            ## eval
            if global_step > start_eval_step and \
                    (global_step - start_eval_step) % eval_batch_step == 0 \
                    and (config['Global']['distributed'] == False or local_rank == 0):
                cur_metric = eval(model,
                                  device,
                                  valid_dataloader,
                                  post_process_class,
                                  eval_class,
                                  model_type,
                                  extra_input=extra_input)
                cur_metric_str = 'cur metric, {}'.format(', '.join(
                    ['{}: {}'.format(k, v) for k, v in cur_metric.items()]))
                logger.info(cur_metric_str)

                # logger metric
                if log_writer is not None:
                    log_metrics(log_writer,
                                metrics=cur_metric,
                                prefix="EVAL",
                                step=global_step)

                if cur_metric[main_indicator] >= best_model_dict[
                        main_indicator]:
                    best_model_dict.update(cur_metric)
                    best_model_dict['best_epoch'] = epoch
                    save_model(model,
                               optimizer,
                               save_model_dir,
                               logger,
                               config,
                               is_best=True,
                               prefix='best_accuracy',
                               best_model_dict=best_model_dict,
                               epoch=epoch,
                               global_step=global_step)
                best_str = 'best metric, {}'.format(', '.join([
                    '{}: {}'.format(k, v) for k, v in best_model_dict.items()
                ]))
                logger.info(best_str)
                # logger best metric
                if log_writer is not None:
                    log_metrics(log_writer,
                                metrics={
                                    "best_{}".format(main_indicator):
                                    best_model_dict[main_indicator]
                                },
                                prefix="EVAL",
                                step=global_step)

            reader_start = time.time()

        if (config['Global']['distributed'] == False or local_rank == 0):
            save_model(model,
                       optimizer,
                       save_model_dir,
                       logger,
                       config,
                       is_best=False,
                       prefix='latest',
                       best_model_dict=best_model_dict,
                       epoch=epoch,
                       global_step=global_step)

        # if (config['Global']['distributed'] == False or dist.get_rank()
        #         == 0) and epoch > 0 and epoch % save_epoch_step == 0:
        #     save_model(model,
        #                optimizer,
        #                save_model_dir,
        #                logger,
        #                config,
        #                is_best=False,
        #                prefix='iter_epoch_{}'.format(epoch),
        #                best_model_dict=best_model_dict,
        #                epoch=epoch,
        #                global_step=global_step)

    profiler.stop()
    
    best_str = 'best metric, {}'.format(', '.join(
        ['{}: {}'.format(k, v) for k, v in best_model_dict.items()]))
    logger.info(best_str)
    if (config['Global']['distributed'] == False
            or local_rank == 0) and log_writer is not None:
        log_writer.close()
    return


def eval(model,
         device,
         valid_dataloader,
         post_process_class,
         eval_class,
         model_type=None,
         extra_input=False):
    model.eval()
    with torch.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(total=len(valid_dataloader),
                    desc='eval model:',
                    position=0,
                    leave=True)
        max_iter = len(valid_dataloader) - 1 if platform.system(
        ) == "Windows" else len(valid_dataloader)
        for idx, batch in enumerate(valid_dataloader):
            gtLabel = [(perData, 1.0) for perData in batch[1]]
            batch = [
                perData.to(device) for perData in batch
                if isinstance(perData, torch.Tensor)
            ]
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()
            if model_type == 'table' or extra_input:
                # preds = model(images, data=batch[1:])
                preds = model(images)
            elif model_type in ["kie", 'vqa']:
                preds = model(batch)
            elif model_type == 'lang':
                preds = model(batch[0], batch[1])
            else:
                preds = model(images)

            batch_numpy = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_numpy.append(item.cpu().numpy())
                else:
                    batch_numpy.append(item)
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            if model_type in ['table', 'kie']:
                eval_class(preds, batch_numpy)
            elif model_type in ['vqa']:
                post_result = post_process_class(preds, batch_numpy)
                eval_class(post_result, batch_numpy)
            elif model_type == 'lang':
                post_result = post_process_class(
                    preds, [batch_numpy[2], batch_numpy[3]])
                eval_class(post_result[0], post_result[1])
            elif model_type == 'det':
                post_result = post_process_class(preds, batch_numpy[1])
                eval_class(post_result, batch_numpy)
            else:
                # post_result = post_process_class(preds, batch_numpy[1])
                post_result = post_process_class(preds)
                if isinstance(preds, dict):
                    for perKey in post_result:
                        post_result[perKey] = [post_result[perKey], gtLabel]
                    eval_class(post_result, batch_numpy)
                else:
                    eval_class([post_result, gtLabel])

            pbar.update(1)
            total_frame += len(images)
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
    model.train()
    metric['fps'] = total_frame / total_time
    return metric


def update_center(char_center, post_result, preds):
    result, label = post_result
    feats, logits = preds
    logits = torch.argmax(logits, dim=-1)
    feats = feats.cpu().numpy()
    logits = logits.cpu().numpy()

    for idx_sample in range(len(label)):
        if result[idx_sample][0] == label[idx_sample][0]:
            feat = feats[idx_sample]
            logit = logits[idx_sample]
            for idx_time in range(len(logit)):
                index = logit[idx_time]
                if index in char_center.keys():
                    char_center[index][0] = (
                        char_center[index][0] * char_center[index][1] +
                        feat[idx_time]) / (char_center[index][1] + 1)
                    char_center[index][1] += 1
                else:
                    char_center[index] = [feat[idx_time], 1]
    return char_center


def get_center(model, eval_dataloader, post_process_class):
    pbar = tqdm(total=len(eval_dataloader), desc='get center:')
    max_iter = len(eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(eval_dataloader)
    char_center = dict()
    for idx, batch in enumerate(eval_dataloader):
        if idx >= max_iter:
            break
        images = batch[0]
        start = time.time()
        preds = model(images)

        batch = [item.cpu.numpy() for item in batch]
        # Obtain usable results from post-processing methods
        post_result = post_process_class(preds, batch[1])

        #update char_center
        char_center = update_center(char_center, post_result, preds)
        pbar.update(1)

    pbar.close()
    for key in char_center.keys():
        char_center[key] = char_center[key][0]
    return char_center


def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profile_dic)

    writer = None
    if is_train:
        # save_config
        save_model_dir = config['Global']['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(dict(config),
                      f,
                      default_flow_style=False,
                      sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
        log_dir = '{}/log'.format(save_model_dir)
        # writer = SummaryWriter(log_dir)
    else:
        log_file = None
        writer = None
    
    distributed = config['Global'].get('distributed', False)
    config['Global']['distributed'] = distributed
    logger = get_logger(log_file=log_file, distributed=distributed)

    use_gpu = config['Global']['use_gpu']
    device = torch.device("cuda" if use_gpu else "cpu")
    check_device(use_gpu)

    print_dict(config, logger)
    logger.info('train with torch {} and device {}'.format(
        torch.__version__, device))
    return config, device, logger, writer
