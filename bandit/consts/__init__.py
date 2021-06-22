from utils import file_functions
from consts.project_paths import ProjectPaths

paths = file_functions.load_json(ProjectPaths.PATHS_CONFIG_PATH)
for name, value in paths.items():
    ProjectPaths.add_path(ProjectPaths, name, value)