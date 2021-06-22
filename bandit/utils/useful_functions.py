import itertools
import pickle
import codecs

from datetime import datetime
import torch
import numpy as np

from utils import sequence_functions


def get_run_name():
    return datetime.now().strftime('%d_%m_%Y_%H_%M_%S')


def format_experiment_name(experiment_name, params):
    for key, value in params.items():
        key_tag = "".join([word[0] for word in key.split('.')[-1].split('_')])
        if isinstance(value, float):
            value = float(value)
            if value.is_integer():
                value = int(value)
        experiment_name += f"_{key_tag}_{value}"
    return experiment_name


def combine_experiment_params(param_ranges):
    def squeeze_experiment_params_attrdict(mapping):
        mapping = dict(mapping)
        top_keys = list(mapping.keys())
        for top_key in top_keys:
            if isinstance(mapping[top_key], dict):
                lower_keys = list(mapping[top_key].keys())
                if 'method' in lower_keys:
                    continue
                for lower_key in lower_keys:
                    mapping['.'.join([top_key, lower_key])] = mapping[top_key][lower_key]
                mapping.pop(top_key)
        dict_exist = False
        for value in mapping.values():
            if isinstance(value, dict) and ('method' not in value):
                dict_exist = True
                break
        if dict_exist:
            mapping = squeeze_experiment_params_attrdict(mapping)
        return mapping

    params = {}
    for config_name, config_params in param_ranges.items():
        config_params = squeeze_experiment_params_attrdict(config_params)
        for param_location, param_data in config_params.items():
            param_location = '.'.join([config_name, param_location])
            if isinstance(param_data, list):
                params[param_location] = param_data
            else:
                params[param_location] = getattr(
                    sequence_functions,
                    param_data.method
                )(**param_data.params)
    param_combinations = [
        dict(zip(params.keys(), [value for value in combination]))
        for combination in itertools.product(*params.values())
    ]
    return param_combinations

        
def base64_encode(tensor):
    encoded = codecs.encode(pickle.dumps(tensor), "base64").decode()
    return encoded


def base64_decode(encode_tensor):
    decoded = pickle.loads(codecs.decode(encode_tensor.encode(), "base64"))
    return decoded


def find_nearest_power_of_two(number):
    start = 1
    while start < number:
        start = start << 1
    start = (start << 1) if ((start - number) < (number >> 1)) else start
    return start


def init_object(module, config, *args, **kwargs):
    if "args" in config:
        args = config.args + args
    if "kwargs" in config:
        kwargs.update(config.kwargs)
    return getattr(module, config.type)(*args, **kwargs)


def is_iterative_object_of_objects(iterative_object, iterative_object_type, inner_object_type):
    if isinstance(iterative_object, iterative_object_type):
        if isinstance(iterative_object_type, dict):
            return all(isinstance(inner_object, inner_object_type)
                       for inner_object in iterative_object.values())
        elif isinstance(iterative_object_type, list):
            return all(isinstance(inner_object, inner_object_type)
                       for inner_object in iterative_object)
    return False


def remove_tensor_wrap(variable):
    """ Convert tensor to numpy and single array to single value  """
    if isinstance(variable, torch.Tensor):
        variable = variable.cpu().detach().numpy()
    if isinstance(variable, np.ndarray) and (variable.size == 1):
        variable = variable.item()
    return variable
