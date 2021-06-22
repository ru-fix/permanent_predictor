from utils import  useful_functions


def __find_best_experiments(self, experiment_locations, experiment_metrics,
                            validation_fucntion_name, best_n = 0, mode = "max", *args, **kwargs):
    # need to return results only for top n best metrics
    results = []
    if self.best_n:
        for experiment_metric in experiment_metrics:
            if self.best_metric in experiment_metric["results"]:
                # call validation function
                results.append(getattr(
                    validation_functions,
                    validation_fucntion_name)(experiment_metric.results[self.best_metric], *args,
                                              **kwargs))
    indices = np.argsort(results)
    # best result is big value or small one?
    indices = indices[::-1] if mode == "max" else indices
    return [experiment_locations[indices[i]] for i in range(best_n)]
