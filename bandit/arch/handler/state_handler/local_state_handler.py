import os
import glob
import pickle

from consts.project_paths import ProjectPaths
from consts.file_names import FileNames
from arch.handler.state_handler.base_state_handler import BaseStateHandler


class LocalStateHandler(BaseStateHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _unpickle_it(file_path):
        pickle_file = open(file_path, "rb")
        result = pickle.load(pickle_file)
        pickle_file.close()
        return result

    @staticmethod
    def _pickle_it(object, file_path):
        pickle_file = open(file_path, "wb")
        pickle.dump(object, pickle_file)
        pickle_file.close()

    @staticmethod
    def __write_it(object, file_path):
        file = open(file_path, 'w')
        file.write(object)
        file.close()

    def save_model_data(self,*subdir_names, **states):
        save_dir_path = os.path.join(ProjectPaths.MODEL_RESULTS_DIR_PATH, *subdir_names)
        os.makedirs(save_dir_path, exist_ok=True)
        for state_name, state in states.items():
            if state_name == "model_state":
                self._pickle_it(state, os.path.join(save_dir_path, FileNames.MODEL))
            elif state_name == "metrics_state":
                self._pickle_it(state, os.path.join(save_dir_path, FileNames.METRICS))
            elif state_name == "datasets_state":
                self._pickle_it(state, os.path.join(save_dir_path, FileNames.DATASETS))
        return save_dir_path

    def load_model_data(self, load_dir_path):
        return self._unpickle_it(os.path.join(load_dir_path, FileNames.MODEL))

    def load_metrics_data(self, load_dir_path):
        return self._unpickle_it(os.path.join(load_dir_path, FileNames.METRICS))

    def load_datasets_data(self, load_dir_path):
        return self._unpickle_it(os.path.join(load_dir_path, FileNames.DATASETS))

    def load_graphics_pickle(self, load_dir_path):
        return self._unpickle_it(os.path.join(load_dir_path, FileNames.GRAPHICS_PICKLE))

    def save_experiment(self, graphics_data, html_data, *subdir_names):
        save_dir_path = os.path.join(ProjectPaths.EXPERIMENT_RESULTS_DIR_PATH, *subdir_names)
        os.makedirs(save_dir_path, exist_ok = True)

        self.__write_it(html_data, os.path.join(save_dir_path, FileNames.GRAPHICS_HTML))
        self._pickle_it(graphics_data, os.path.join(save_dir_path, FileNames.GRAPHICS_PICKLE))

    def format_experiment_location(self, dirpaths):
        locations = []
        for dirpath in dirpaths:
            if dirpath.endswith("*"):
                if os.path.exists(os.path.split(dirpath)[0]):
                    locations += glob.glob(dirpath)
                else:
                    locations += glob.glob(
                        os.path.join(ProjectPaths.EXPERIMENT_RESULTS_DIR_PATH, dirpath)
                    )
            else:
                if os.path.exists(dirpath):
                    locations.append(dirpath)
                else:
                    locations.append(os.path.join(
                        ProjectPaths.EXPERIMENT_RESULTS_DIR_PATH, dirpath)
                    )
        return locations

    def save_analyze(self, html_data, *subdir_names):
        save_dir_path = os.path.join(ProjectPaths.ANALYZE_RESULTS_DIR_PATH, *subdir_names)
        os.makedirs(save_dir_path, exist_ok=True)

        self.__write_it(html_data, os.path.join(save_dir_path, FileNames.GRAPHICS_HTML))

    def load_experiments_graphics(self, experiments_location):
        if os.path.exists(experiments_location):
            experiments_graphics_paths = glob.glob(experiments_location)
        else:
            experiments_graphics_paths = glob.glob(os.path.join(
                ProjectPaths.EXPERIMENT_RESULTS_DIR_PATH,
                experiments_location
            ))

        experiments_graphics = {}
        for experiment_graphics_path in experiments_graphics_paths:
            experiment_name = os.path.split(experiment_graphics_path)[1]
            experiments_graphics[experiment_name] = self.load_graphics_pickle(experiment_graphics_path)

        return experiments_graphics