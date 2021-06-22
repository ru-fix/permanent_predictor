import argparse

from arch import analyzer_archs, handler_archs
from utils.config_parser import ConfigParser
from utils import useful_functions
from consts.project_paths import ProjectPaths


def main(config):
    run_name = useful_functions.get_run_name()
    for analyze_name, analyze_config in config.analyzes.items():
        state_handler = useful_functions.init_object(
            handler_archs,
            config.handlers[analyze_config.handler],
        )
        analyzer = useful_functions.init_object(
            analyzer_archs,
            config.analyzers[analyze_config.analyzer],
            state_handler=state_handler,
            **analyze_config.kwargs
        )
        getattr(analyzer, analyze_config.type)(analyze_name, run_name, **analyze_config.kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration files")
    parser.add_argument(
        '-analyzes',
        default=ProjectPaths.ANALYZE_DEFAULT_CONFIG_PATH,
        nargs='?',
        help='Experiments configuration file path',
    )
    parser.add_argument(
        '-additionals',
        default = [
            ProjectPaths.ANALYZER_CONFIGS_DIR_PATH,
            ProjectPaths.HANDLER_CONFIGS_DIR_PATH
        ],
        nargs='+',
        help = 'Additional configuration file paths',
    )
    args = vars(parser.parse_args())

    config = ConfigParser()
    config.load(args)
    main(config)
