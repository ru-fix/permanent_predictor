"""
Base trainer for all trainers
"""

from utils import useful_functions, sequence_functions


class BaseTrainer:
    def __init__(self, last_save_description=None, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.last_save_description = last_save_description
        if self.last_save_description is not None:
            self.load()

        if hasattr(self, 'amount_rounds'):
            self.__calculate_amount_possible_rounds()
            self.__calculate_metrics_save_period()

        if hasattr(self, 'metrics'):
            self.__format_save_metric_variable_names()

    def __calculate_amount_possible_rounds(self):
        # Input amount_rounds range: (1, amount_contexts) (user set it)
        # 1. if amount_rounds is one with exploration its mean
        #   that after exploration phase will one train iteration
        # 2. if amount rounds is one without exploration ite mean
        #   that all contexts will user at inference without train iterations

        self.exploration_exist = hasattr(self.model, 'exploration_trials')

        # Get amount exploitation trials
        amount_exploitation_contexts = self.environment.amount_contexts
        # if exploitation exist
        if self.exploration_exist:
            # Reduce exploitation trials
            amount_exploitation_contexts -= self.model.exploration_trials
        # Calculate amount exploitation trials in round
        amount_round_exploitation_trials = amount_exploitation_contexts / self.amount_rounds
        # If relation is less than one then desire rounds > possible rounds
        if amount_round_exploitation_trials < 1:
            # Set amount rounds as amount context
            self.amount_rounds = amount_exploitation_contexts

        # Sequence function input amount_rounds range:
        # 1. If exploration exist (1, amount_contexts + 1)
        # 2. If exploration not exist (2, amount_contexts + 1)
        #   (because first point is zero will be skip)
        if self.exploration_exist:
            self.round_iterations = getattr(sequence_functions, self.round_iterations_rule)(
                start = self.model.exploration_trials,
                stop = self.environment.amount_contexts,
                num = self.amount_rounds,
                dtype = int
            )
        else:
            self.round_iterations = getattr(sequence_functions, self.round_iterations_rule)(
                start = 0,
                stop = self.environment.amount_contexts,
                num = self.amount_rounds + 1,
                dtype = int
            )[1:]

    def __format_save_metric_variable_names(self):
        self.save_metric_variable_names = []
        for metric in self.metrics.values():
            self.save_metric_variable_names += metric.arg_names
            if ('kwargs' in metric) and ('split' in metric.kwargs):
                self.save_metric_variable_names.append(metric.kwargs.split)
        self.save_metric_variable_names = set(self.save_metric_variable_names)

    def __calculate_metrics_save_period(self):
        self.metrics_save_period = max(1, int(self.amount_rounds / self.amount_save_rounds))

    def calculate_metrics(self):
        for name, data in self.metrics.items():
            self.metrics_handler.calculate_metric(
                name,
                data.type,
                data.arg_names,
                **data.kwargs if "kwargs" in data else {}
            )

    def get_last_save_description(self):
        return self.last_save_description

    def save(self, *save_desc):
        self.last_save_description = self.state_handler.save_model_data(
            self.experiment_name,
            self.run_name,
            *save_desc,
            model_state=self.model.state_dict(),
            metrics_state=self.metrics_handler.get_data(),
            datasets_state = self.datasets_handler.get_data()
        )

    def load(self):
        self.model.load(self.state_handler.load_model_data(self.last_save_description))
        self.metrics_handler.load(self.handler.load_metrics_data(self.last_save_description))
        self.datasets_handler.load(self.handler.load_datasets_data(self.last_save_description))

    def __save_samples(self, iteration_variables):
        self.datasets_handler.add(
            *[useful_functions.remove_tensor_wrap(iteration_variables[name])
              for name in self.save_dataset_variable_names]
        )

    def __save_metrics_data(self, iteration_variables):
        for name in iteration_variables:
            if name in self.save_metric_variable_names:
                self.metrics_handler.add_values(
                    name,
                    useful_functions.remove_tensor_wrap(iteration_variables[name])
                )

    def _iteration_decorator(formate_args=False, save_samples=True, save_metrics=True):
        def top_wrapper(iteration_function):
            def bottom_wrapper(self, *args, iteration_desc=None, **kwargs):
                if formate_args:
                    args = [useful_functions.remove_tensor_wrap(arg) for arg in args]
                    if iteration_desc is not None:
                        iteration_desc = {
                            key: useful_functions.remove_tensor_wrap(value)
                            for key, value in iteration_desc.items()
                        }
                    kwargs = {
                        key: useful_functions.remove_tensor_wrap(value)
                        for key, value in kwargs.items()
                    }
                iteration_variables = iteration_function(self, *args, **kwargs)
                if iteration_desc is not None:
                    iteration_variables.update(iteration_desc)
                if save_samples:
                    self.__save_samples(iteration_variables)
                if save_metrics:
                    self.__save_metrics_data(iteration_variables)
            return bottom_wrapper
        return top_wrapper

    def train(self):
        raise NotImplementedError('Not implement method \'train\'')
