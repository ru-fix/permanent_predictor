import argparse
import gym

from arch import model_archs, trainer_archs, analyzer_archs, handler_archs
from utils.config_parser import ConfigParser
from utils import useful_functions
from consts.project_paths import ProjectPaths


def run(config, experiment_config, run_name, experiment_name):
    environment = useful_functions.init_object(
        gym,
        config.environments[experiment_config.environment]
    )
    model = useful_functions.init_object(
        model_archs,
        config.models[experiment_config.model]
    )
    metrics_handler = handler_archs.MetricsHandler(experiment_name, run_name)
    datasets_handler = useful_functions.init_object(
        handler_archs,
        config.handlers[experiment_config.datasets_handler]
    )
    state_handler = useful_functions.init_object(
        handler_archs,
        config.handlers[experiment_config.state_handler],
    )

    trainer = useful_functions.init_object(
        trainer_archs,
        config.trainers[experiment_config.trainer],
        model=model,
        environment=environment,
        metrics_handler=metrics_handler,
        datasets_handler=datasets_handler,
        state_handler=state_handler,
        experiment_name=experiment_name,
        run_name=run_name
    )
    last_model_save_path = trainer.train()

    analyzer = useful_functions.init_object(
        analyzer_archs,
        config.analyzers[experiment_config.analyzer],
        state_handler=state_handler,
    )
    analyzer.build(last_model_save_path)


def main(config):
    run_name = useful_functions.get_run_name()
    for experiment_name, experiment_config in config.experiments.items():
        if experiment_config.run_params is None:
            run(config, experiment_config, run_name, experiment_name)
        else:
            param_combinations = \
                useful_functions.combine_experiment_params(experiment_config.run_params)
            for param_combination in param_combinations:
                config.update(**param_combination)
                run(config,
                    experiment_config,
                    run_name,
                    useful_functions.format_experiment_name(experiment_name, param_combination))


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration files")
    parser.add_argument(
        '-experiments',
        default=ProjectPaths.EXPERIMENT_DEFAULT_CONFIG_PATH,
        help='Experiments configuration file path',
    )
    parser.add_argument(
        '-additionals',
        default = [
            ProjectPaths.ENVIRONMENT_CONFIGS_DIR_PATH,
            ProjectPaths.HANDLER_CONFIGS_DIR_PATH,
            ProjectPaths.MODEL_CONFIGS_DIR_PATH,
            ProjectPaths.TRAINERS_CONFIGS_DIR_PATH,
            ProjectPaths.ANALYZER_CONFIGS_DIR_PATH,

        ],
        nargs='+',
        help='Additional configuration file paths',
    )
    args = vars(parser.parse_args())

    config = ConfigParser()
    config.load(args)
    main(config)