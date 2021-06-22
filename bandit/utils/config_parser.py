import os
import glob

from utils.attrdict import AttrDict
from utils import file_functions


class ConfigParser:
    def __init__(self):
        self.configs = AttrDict()

    def __getattr__(self, key):
        return self.configs.__getattr__(key)
    
    def __getitem__(self, key):
        return self.configs.__getitem__(key)

    def __is_location(self, arg, location):
        if arg in self[location]:
            return True, f"{location}.{arg}"
        elif arg in self.configs:
            return True, arg
        else:
            return False, None

    def __get_arg_name(self, arg):
        if isinstance(arg, str):
            return arg.split('.')[-1]
        else:
            return None

    def __add_arg_value(self, args, name, value, args_location):
        star_exist = False

        if isinstance(value, str):
            if value.endswith("*"):
                star_exist = True
                value = value[:-1]
            is_location, location = self.__is_location(value, args_location)
            if is_location:
                value = self[location]
            elif star_exist:
                star_exist = False
                value += '*'

        if name in args:
            if isinstance(value, dict):
                if star_exist:
                    args.update(value)
                else:
                    args[name].update(value)
            else:
                raise TypeError(f"Can not update {type(args[name])} "
                                f"with {type(value)} element")
        else:
            if star_exist:
                if isinstance(value, dict):
                    args.update(value)
                else:
                    raise TypeError(f"Can not add {type(value)} with * element")
            else:
                args[name] = value


    def __parse_args(self, raw_args, args_location, add_key=True):
        args = dict()
        if isinstance(raw_args, list):
            for value in raw_args:
                self.__add_arg_value(args, self.__get_arg_name(value), value, args_location)
        elif isinstance(raw_args, dict):
            for name, value in raw_args.items():
                self.__add_arg_value(args, name, value, args_location)
        else:
            self.__add_arg_value(args, self.__get_arg_name(raw_args), raw_args, args_location)

        if not add_key:
            return tuple(args.values())

        return args

    def update(self, *args, **kwargs):
        self.configs.update(*args, **kwargs)

    def __prepare_args_and_kwargs(self, config, config_location):
        for param_name, param_data in config.items():
            if isinstance(param_data, dict):
                self.__prepare_args_and_kwargs(param_data, config_location)
            if param_name.endswith("kwargs"):
                config[param_name] = self.__parse_args(param_data, config_location)
            elif param_name.endswith("args"):
                config[param_name] = self.__parse_args(param_data, config_location, add_key=False)

    def load(self, args):
        config_locations = []
        for arg_value in args.values():
            if isinstance(arg_value, str):
                config_locations.append(arg_value)
            else:
                config_locations.extend(arg_value)

        for config_location in config_locations:
            if os.path.isdir(config_location):
                config_file_paths = sum([glob.glob(os.path.join(dir, '*.json'))
                                         for dir in glob.glob(os.path.join(config_location, '*/'))], [])
                config_file_paths += glob.glob(
                    os.path.join(config_location, '*.json'),
                )
                for config_file_path in config_file_paths:
                    config = file_functions.load_json(config_file_path)
                    self.configs.update(**config)
            else:
                config = file_functions.load_json(config_location)
                self.configs.update(**config)

        for category_name, configs in self.configs.items():
            for config_name, config_data_data in configs.items():
                self.__prepare_args_and_kwargs(config_data_data, f"{category_name}.{config_name}")
