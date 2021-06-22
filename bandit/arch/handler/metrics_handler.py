from arch.model import metrics
from utils.attrdict import AttrDict


class MetricsHandler:
    def __init__(self, experiment_name, run_name):
        self.metrics = AttrDict({
            "data": {},
            "results": {},
            "params": {},
            "experiment_name": experiment_name,
            "run_name": run_name
        })

    def add_values(self, value_name, value):
        """
        Saves next value of 'value_name' feature internally.
        :param value_name: str. Variable name as in source code.
        :param value: Object. Value to save internally.
        :return: None. Updates self.values with metric input data.
        """
        if value_name not in self.metrics.data:
            self.metrics.data[value_name] = []
        self.metrics.data[value_name].append(value)

    def _replace(self, metric_name, metric_result, split):
        if split is None:
            self.metrics.results[metric_name] = metric_result
        else:
            if metric_name not in self.metrics.results:
                self.metrics.results[metric_name] = {}
            split_value = self.metrics.data[split][-1]
            self.metrics.results[metric_name][split_value] = metric_result

    def _update(self, metric_name, metric_result, split):
        if split is None:
            if metric_name not in self.metrics.results:
                self.metrics.results[metric_name] = metric_result
            else:
                if isinstance(metric_result, dict):
                    self.metrics.results[metric_name].update(metric_result)
        else:
            if metric_name not in self.metrics.results:
                self.metrics.results[metric_name] = {}
            split_value = self.metrics.data[split][-1]
            if split_value not in self.metrics.results[metric_name]:
                self.metrics.results[metric_name][split_value] = metric_result
            else:
                if isinstance(metric_result, dict):
                    self.metrics.results[metric_name][split_value].update(metric_result)

    def _append(self, metric_name, metric_result, split):
        if split is None:
            if metric_name not in self.metrics.results:
                self.metrics.results[metric_name] = []
            self.metrics.results[metric_name].append(metric_result)
        else:
            if metric_name not in self.metrics.results:
                self.metrics.results[metric_name] = {}
            split_value = self.metrics.data[split][-1]
            if split_value not in self.metrics.results[metric_name]:
                self.metrics.results[metric_name][split_value] = []
            self.metrics.results[metric_name][split_value].append(metric_result)

    def calculate_metric(self, metric_name, func_name, arg_names,
                         split=None, mode='append', **kwargs):
        """
        Forms arg_names list of arguments to pass into 'name' function metric from lib
        to calculate metric.
        :param metric_name: str. Function name from metrics lib as in source code.
        :param arg_names: Iterable object. Ex, list of strings with args names as in source code.
        :return: None. Updates self.results with calculated metric.
        """
        args = []
        for arg_name in arg_names:
            args.append(self.metrics.data[arg_name])
        metric_result = getattr(metrics, func_name)(*args, **kwargs)

        if kwargs:
            self.metrics.params[metric_name] = kwargs

        getattr(self, f"_{mode}")(metric_name, metric_result, split)

    def get_data(self):
        return self.metrics

    def load(self, metrics_data):
        self.metrics = metrics_data