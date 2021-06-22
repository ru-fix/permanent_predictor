from plotly.graph_objects import Figure

from arch.analyzer.utils.graphic_container import GraphicContainer
from consts import html_bases

class BaseAnalyzer:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __format_graphics(self, metrics):
        graphics = {}
        for metric_name, metric_data in metrics.results.items():
            if metric_name in self.graphics_params:
               graphics[metric_name] = GraphicContainer(
                   metric_data,
                   self.graphics_params[metric_name],
                   self.default_graphics_params,
                   metrics.params[metric_name],
                   metric_name,
                   metrics.experiment_name
               )
        return graphics

    def __recursive_dropdown_build(self, current_key, key, dropdown_item, graphic_name):
        # leaf nodes
        if isinstance(dropdown_item, Figure):
            return current_key, key, html_bases.wrap_into_div(
                graphic_name+current_key,
                dropdown_item.to_html(
                    include_plotlyjs=False,
                    full_html=False))

        # recursion
        current_level_dropdown_ids = []
        current_level_dropdown_keys = []
        current_level_dropdown_items = []
        for index, (key, content) in enumerate(dropdown_item.items()):
            full_ids, level_ids, item_htmls = self.__recursive_dropdown_build(
                current_key + "_" + key,
                key,
                content,
                graphic_name
            )
            # inner result
            if isinstance(full_ids, list):
                # we need all full ids
                current_level_dropdown_ids += full_ids
                if index == 0:
                    level_ids.append(list(dropdown_item.keys()))
                    current_level_dropdown_keys = level_ids
            # leaf result
            else:
                # get only first child info
                if index == 0:
                    current_level_dropdown_ids = [[current_key + "_" + key for key in dropdown_item.keys()]]
                    current_level_dropdown_keys = [[key for key in dropdown_item.keys()]]


            current_level_dropdown_items.append(item_htmls)

        return current_level_dropdown_ids, current_level_dropdown_keys, "\n".join(current_level_dropdown_items)

    def __build_html(self, graphics):
        # creating script
        html_script = html_bases.wrap_into_script(
            "\n".join(base.get_html_element_script()
                      for base in [html_bases.Script, html_bases.Buttons, html_bases.Dropdown])
        )
        html_style = html_bases.Style.get_html_element_base()
        graphic_group_prefix = "graphic_group"
        graphic_groups = [f"{graphic_group_prefix}_{key}" for key in graphics.keys()]
        html_body = [html_bases.Buttons.get_html_element_base(graphic_groups, graphics.keys(), graphic_group_prefix)]
        for graphic_group_index, (graphic_name, graphic) in enumerate(graphics.items()):
            div_graphic_id = f"{graphic_name}_graph"

            content = graphic.get()
            if isinstance(content, dict):
                dropdown_full_ids, dropdowns_values, graphic_html = self.__recursive_dropdown_build("", "", content, div_graphic_id)
                dropdowns_values.reverse()

                dropdowns = []
                for index, dropdown_type_values in enumerate(dropdowns_values):
                    dropdown_id = f"{div_graphic_id}_{index}"
                    dropdown = html_bases.Dropdown.get_html_element_base(dropdown_type_values, dropdown_id, div_graphic_id)
                    dropdowns.append(dropdown)

                dropdowns = "\n".join(dropdowns)

                graphic_html = '\n'.join([dropdowns, graphic_html])
            else:
                graphic_html = content.to_html(
                    include_plotlyjs=False,
                    full_html=False
                )
            html_body.append(html_bases.wrap_into_div(graphic_groups[graphic_group_index], graphic_html))
        result_html_body = html_bases.wrap_into_body('\n'.join(html_body))
        result_head = html_bases.wrap_into_head('\n'.join([html_style, html_script]))
        result_html = html_bases.wrap_into_html('\n'.join([result_head, result_html_body]))
        return result_html

    def build(self, metrics_path):
        metrics = self.state_handler.load_metrics_data(metrics_path)
        graphics = self.__format_graphics(metrics)
        html = self.__build_html(graphics)
        self.state_handler.save_experiment(graphics, html, metrics.run_name, metrics.experiment_name)

    def compare_experiments(self, analyze_name, run_name, experiment_locations):
        experiments_graphics = self.state_handler.load_experiments_graphics(experiment_locations)

        compare_graphics = {}
        for experiment_name, experiment_graphics in experiments_graphics.items():
            for metric_name, experiment_graphic in experiment_graphics.items():
                if metric_name not in compare_graphics:
                    experiment_graphic.update_graphic_params(
                        self.default_compare_graphics_params
                    )
                    if metric_name in self.compare_graphics_params:
                        experiment_graphic.update_graphic_params(
                            self.compare_graphics_params[metric_name]
                        )
                    compare_graphics[metric_name] = experiment_graphic
                else:
                    compare_graphics[metric_name].add(experiment_name, experiment_graphic)

        html = self.__build_html(compare_graphics)
        self.state_handler.save_analyze(html, analyze_name, run_name)

