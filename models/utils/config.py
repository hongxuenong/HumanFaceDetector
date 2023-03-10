import yaml
import os


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


def add_config_rec(config, char_num):
    if config['Architecture']["algorithm"] in [
            "Distillation",
    ]:  # distillation model
        for key in config['Architecture']["Models"]:
            if config['Architecture']['Models'][key]['Head'][
                    'name'] == 'MultiHead':  # for multi head
                if config['PostProcess'][
                        'name'] == 'DistillationSARLabelDecode':
                    char_num = char_num - 2
                # update SARLoss params
                assert list(config['Loss']['loss_config_list']
                            [-1].keys())[0] == 'DistillationSARLoss'
                config['Loss']['loss_config_list'][-1]['DistillationSARLoss'][
                    'ignore_index'] = char_num + 1
                out_channels_list = {}
                out_channels_list['CTCLabelDecode'] = char_num
                out_channels_list['SARLabelDecode'] = char_num + 2
                config['Architecture']['Models'][key]['Head'][
                    'out_channels_list'] = out_channels_list
            else:
                config['Architecture']["Models"][key]["Head"][
                    'out_channels'] = char_num
    elif config['Architecture']['Head'][
            'name'] == 'MultiHead':  # for multi head
        if config['PostProcess']['name'] == 'SARLabelDecode':
            char_num = char_num - 2
        # update SARLoss params
        assert list(
            config['Loss']['loss_config_list'][1].keys())[0] == 'SARLoss'
        if config['Loss']['loss_config_list'][1]['SARLoss'] is None:
            config['Loss']['loss_config_list'][1]['SARLoss'] = {
                'ignore_index': char_num + 1
            }
        else:
            config['Loss']['loss_config_list'][1]['SARLoss'][
                'ignore_index'] = char_num + 1
        out_channels_list = {}
        out_channels_list['CTCLabelDecode'] = char_num
        out_channels_list['SARLabelDecode'] = char_num + 2
        config['Architecture']['Head']['out_channels_list'] = out_channels_list
    else:  # base rec model
        config['Architecture']["Head"]['out_channels'] = char_num

    if config['PostProcess']['name'] == 'SARLabelDecode':  # for SAR model
        config['Loss']['ignore_index'] = char_num - 1
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
