import os

class ProjectPaths:
    # Main project paths
    PROJECT_DIR_PATH = os.path.split(os.path.dirname(__file__))[0]

    # Project directory paths
    CONFIG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'configs')
    RESOURCES_DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'resources')


    # Config directory paths
    MODEL_CONFIGS_DIR_PATH = os.path.join(CONFIG_DIR_PATH, 'models')
    TRAINERS_CONFIGS_DIR_PATH = os.path.join(CONFIG_DIR_PATH, 'trainers')
    ENVIRONMENT_CONFIGS_DIR_PATH = os.path.join(CONFIG_DIR_PATH, 'environments')
    EXPERIMENT_CONFIGS_DIR_PATH = os.path.join(CONFIG_DIR_PATH, 'experiments')
    ANALYZE_CONFIGS_DIR_PATH = os.path.join(CONFIG_DIR_PATH, 'analyzes')
    ANALYZER_CONFIGS_DIR_PATH = os.path.join(CONFIG_DIR_PATH, 'analyzers')
    HANDLER_CONFIGS_DIR_PATH = os.path.join(CONFIG_DIR_PATH, 'handlers')


    # Config paths
    PATHS_CONFIG_PATH = os.path.join(CONFIG_DIR_PATH, 'paths.json')
    EXPERIMENT_DEFAULT_CONFIG_PATH = os.path.join(EXPERIMENT_CONFIGS_DIR_PATH, 'bandit.json')
    ANALYZE_DEFAULT_CONFIG_PATH = os.path.join(ANALYZE_CONFIGS_DIR_PATH, 'default.json')

    def add_path(self, name, value):
        setattr(self, name, value)
