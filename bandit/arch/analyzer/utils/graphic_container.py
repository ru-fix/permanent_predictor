from plotly.graph_objects import Figure as Graphic

from arch.analyzer.graphical_objects import go_archs, union_functions


class GraphicContainer:
    def __init__(self, graphic_data, graphic_params, default_graphic_params, metric_params, metric_name, experiment_name):
        self.graphic_params = default_graphic_params.clone()
        self.graphic_params.update(graphic_params)
        self.__save_experiment_name(experiment_name)
        self.metric_params = metric_params
        if metric_params is not None:
            self.graphic_params.layout.title = \
                self.graphic_params.layout.title.format(**self.metric_params)

        self.metric_name = metric_name

        self.go = getattr(go_archs, self.graphic_params.type)(
            graphic_data,
            x_axis_mode=self.graphic_params.x_axis_mode
        )

    def __save_experiment_name(self, experiment_name):
        if self.graphic_params.data is None:
            self.graphic_params.data = [{"name": experiment_name}]
        else:
            self.graphic_params.data.append({"name": experiment_name})

    def __get_graphic(self, go_data):
        graphic = Graphic(go_data)
        graphic.update_layout(self.graphic_params.layout)
        graphic.update_traces(self.graphic_params.traces)
        if len(go_data) == len(self.graphic_params.data):
            graphic.update(data=self.graphic_params.data)
        return graphic

    def __recursive_get_graphic(self, go_data, level=0):
        if isinstance(go_data, dict):
            sub_dict = {}
            for sub_name, sub_go_data in go_data.items():
                key = f"{self.graphic_params.group_item_names[level]}_{sub_name}"
                sub_dict[key] = self.__recursive_get_graphic(sub_go_data, level+1)
            return sub_dict
        else:
            return self.__get_graphic(go_data)

    def get(self):
        if isinstance(self.go, go_archs.BaseGraphicalObjectGroup):
            return self.__recursive_get_graphic(self.go.get())
        else:
            return self.__get_graphic(self.go.get())

    def update_graphic_params(self, new_graphic_params):
        self.graphic_params.update(new_graphic_params)

    def add(self, experiment_name, new_graphic_container):
        self.go = union_functions.union(self.go, new_graphic_container.go)
        self.__save_experiment_name(experiment_name)
