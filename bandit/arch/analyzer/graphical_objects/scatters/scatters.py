from arch.analyzer.graphical_objects.base_go import BaseGraphicalObjectGroup
from arch.analyzer.graphical_objects.scatters.base_scatters import BaseScatter, BaseMultiScatter


class Scatter(BaseScatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preprocess_data()

    def _preprocess_data(self):
        if self.x_axis_mode == 'exist':
            if len(self.data) == 2:
                self.data = dict(
                    x=self.data[0],
                    y=self.data[1],
                )
            else:
                self.data = dict(
                    x=[value[0] for value in self.data],
                    y=[value[1] for value in self.data],
                )
        else:
            self.data = dict(
                x=list(range(len(self.data))),
                y=self.data,
            )


class MultiScatter(BaseMultiScatter):
    def __init__(self, *args, preprocess=True, **kwargs):
        super().__init__(*args, **kwargs)
        if preprocess:
            self._preprocess_data()

    def _preprocess_data(self):
        """

        :param raw_data: List of dicts
        :return:
        """

        format_data = {}
        if self.x_axis_mode is None:
            for data_dict in self.data:
                for data_dict_key, data_dict_value in data_dict.items():
                    if data_dict_key not in format_data:
                        format_data[data_dict_key] = dict(
                            y=[],
                            name=f"{data_dict_key}"
                        )
                    format_data[data_dict_key]['y'].append(data_dict_value)
            for data_dict_key in format_data:
                format_data[data_dict_key]['x'] = list(range(len(format_data[data_dict_key]['y'])))
        else:
            for index, data_dict in enumerate(self.data):
                for data_dict_key, data_dict_value in data_dict.items():
                    if data_dict_key not in format_data:
                        format_data[data_dict_key] = dict(
                            x = [],
                            y = [],
                            name = f"{data_dict_key}"
                        )
                    if self.x_axis_mode == 'exist':
                        format_data[data_dict_key]['x'].append(data_dict_value[0])
                        format_data[data_dict_key]['y'].append(data_dict_value[1])
                    elif self.x_axis_mode == 'async':
                        format_data[data_dict_key]['x'].append(index)
                        format_data[data_dict_key]['y'].append(data_dict_value)

        self.data = list(format_data.values())


class MultiScatterGroups(BaseGraphicalObjectGroup):
    def __init__(self, *args, preprocess=True, **kwargs):
        super().__init__(*args, **kwargs)
        if preprocess:
            self._preprocess_data()

    def _recursive_preprocess_data(self, data):
        if isinstance(data, dict):
            return {
                key: self._recursive_preprocess_data(subdata)
                for key, subdata in data.items()
            }
        else:
            return MultiScatter(data, **self.go_kwargs).get()
