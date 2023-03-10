import torch
import os
import errno
import pickle
from .init import weight_init
from .logging import get_logger


def load_model(config,
               model,
               device,
               logger,
               optimizer=None):
    """
    load model from checkpoint or pretrained_model
    """
    # logger = get_logger()
    # if not config['Global']['checkpoints'] and not config['Global']['pretrained_model']:
    #     model.apply(weight_init)
    global_config = config['Global']
    checkpoints = global_config.get('checkpoints')
    pretrained_model = global_config.get('pretrained_model')
    best_model_dict = {}

    # if model_type == 'vqa':
    #     checkpoints = config['Architecture']['Backbone']['checkpoints']
    #     # load vqa method metric
    #     if checkpoints:
    #         if os.path.exists(os.path.join(checkpoints, 'metric.states')):
    #             with open(os.path.join(checkpoints, 'metric.states'),
    #                     'rb') as f:
    #                 states_dict = pickle.load(f) if six.PY2 else pickle.load(
    #                     f, encoding='latin1')
    #             best_model_dict = states_dict.get('best_model_dict', {})
    #             if 'epoch' in states_dict:
    #                 best_model_dict['start_epoch'] = states_dict['epoch'] + 1
    #         logger.info("resume from {}".format(checkpoints))

    #         if optimizer is not None:
    #             if checkpoints[-1] in ['/', '\\']:
    #                 checkpoints = checkpoints[:-1]
    #             if os.path.exists(checkpoints + '.pdopt'):
    #                 optim_dict = paddle.load(checkpoints + '.pdopt')
    #                 optimizer.set_state_dict(optim_dict)
    #             else:
    #                 logger.warning(
    #                     "{}.pdopt is not exists, params of optimizer is not loaded".
    #                     format(checkpoints))
    #     return best_model_dict

    if checkpoints:
        if checkpoints.endswith('.ptparams'):
            checkpoints = checkpoints.replace('.ptparams', '')
        assert os.path.exists(checkpoints + ".ptparams"), \
            "The {}.ptparams does not exists!".format(checkpoints)

        # load params from trained model
        params = torch.load(checkpoints + '.ptparams', map_location=device)
        state_dict = model.state_dict()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key not in params:
                logger.warning("{} not in loaded params {} !".format(
                    key, params.keys()))
                continue
            pre_value = params[key]
            if list(value.shape) == list(pre_value.shape):
                new_state_dict[key] = pre_value
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params shape {} !"
                    .format(key, value.shape, pre_value.shape))
        model.load_state_dict(new_state_dict)

        if optimizer is not None:
            if os.path.exists(checkpoints + '.ptopt'):
                optim_dict = torch.load(checkpoints + '.ptopt',
                                        map_location=device)
                optimizer.load_state_dict(optim_dict)
            else:
                logger.warning(
                    "{}.ptopt is not exists, params of optimizer is not loaded"
                    .format(checkpoints))

        if os.path.exists(checkpoints + '.states'):
            with open(checkpoints + '.states', 'rb') as f:
                states_dict = pickle.load(f, encoding='latin1')
            best_model_dict = states_dict.get('best_model_dict', {})
            if 'epoch' in states_dict:
                best_model_dict['start_epoch'] = states_dict['epoch'] + 1
        logger.info("resume from {}".format(checkpoints))
    elif pretrained_model:
        load_pretrained_params(model, pretrained_model, logger)
    else:
        logger.info('train from scratch')
    return best_model_dict


def load_pretrained_params(model, path, logger=None):
    if not logger:
        logger = get_logger()
    if path.endswith('.ptparams'):
        path = path.replace('.ptparams', '')
    assert os.path.exists(path + ".ptparams"), \
        "The {}.ptparams does not exists!".format(path)

    params = torch.load(path + '.ptparams')

    ## load params to dis setting yuchi
    # params_dis = {}
    # for k1 in params.keys():
    #     params_dis["Teacher."+k1] = params[k1]
    #     params_dis["Student."+k1] = params[k1]
    # params = params_dis

    ## only load backbone
    # params_dis = {}
    # for k1 in params.keys():
    #     if "Backbone" in k1:
    #         params_dis[k1] = params[k1]
    # params = params_dis

    state_dict = model.state_dict()
    new_state_dict = {}
    for k1 in params.keys():
        if k1 not in state_dict.keys():
            logger.warning("The pretrained params {} not in model".format(k1))
        else:
            if list(state_dict[k1].shape) == list(params[k1].shape):
                new_state_dict[k1] = params[k1]
            else:
                logger.warning(
                    "The shape of model params {} {} not matched with loaded params {} {} !"
                    .format(k1, state_dict[k1].shape, k1, params[k1].shape))
    if len(state_dict) > len(new_state_dict):  # load a part of weights
        for k in state_dict.keys():
            if k not in new_state_dict.keys():
                new_state_dict[k] = state_dict[k]
    model.load_state_dict(new_state_dict)
    logger.info("load pretrain successful from {}".format(path))
    return model


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def save_model(model,
               optimizer,
               model_path,
               logger,
               config,
               is_best=False,
               prefix='tritonocr',
               **kwargs):
    """
    save model to the target path
    """
    _mkdir_if_not_exist(model_path, logger)
    model_prefix = os.path.join(model_path, prefix)
    torch.save(optimizer.state_dict(), model_prefix + '.ptopt')
    if config['Architecture']["model_type"] != 'vqa':
        torch.save(model.state_dict(), model_prefix + '.ptparams')
        metric_prefix = model_prefix
    else:
        if config['Global']['distributed']:
            model._layers.backbone.model.save_pretrained(model_prefix)
        else:
            model.backbone.model.save_pretrained(model_prefix)
        metric_prefix = os.path.join(model_prefix, 'metric')
    # save metric and config
    with open(metric_prefix + '.states', 'wb') as f:
        pickle.dump(kwargs, f, protocol=2)
    if is_best:
        logger.info('save best model is to {}'.format(model_prefix))
    else:
        logger.info("save model in {}".format(model_prefix))
