

class BaseStateHandler:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_model_data(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'save_model_data\'')

    def save_metric_data(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'save_metric_data\'')

    def save_dataset_data(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'save_dataset_data\'')

    def load_model_data(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'load_model_data\'')

    def load_metric_data(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'load_metric_data\'')

    def load_dataset_data(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'load_dataset_data\'')

    def load_format_metrics(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'load_format_metrics\'')

    def save_experiment(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'save_experiment\'')

    def format_experiment_location(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'format_experiment_location\'')

    def save_analyze(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'save_analyze\'')

    def load_experiments_layouts(self, *args, **kwargs):
        raise NotImplementedError('Not implement method \'load_experiments_layouts\'')