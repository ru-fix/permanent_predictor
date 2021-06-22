import os

from consts.project_paths import ProjectPaths
from consts.file_names import FileNames


def wrap_into_div(element_name, element_content):
    return f"<div id=\"{element_name}\">\n\t{element_content}\n</div>"

def wrap_into_script(element_content):
    return f"<script>\n{element_content}\n</script>"

def wrap_into_html(element_content):
    return f"<html>\n{element_content}\n</html>"

def wrap_into_head(element_content):
    return f"<head>\n{element_content}\n</head>"

def wrap_into_body(element_content):
    return f"<body>\n{element_content}\n</body>"


class Style():
    style_item = ""

    with open(os.path.join(ProjectPaths.RESOURCES_DIR_PATH, FileNames.STYLES), 'r') as f:
        style_item = f"<style>\n{f.read()}\n</style>"

    @staticmethod
    def get_html_element_base():
        return Style.style_item

class Script():
    script_item = ""
    with open(os.path.join(ProjectPaths.RESOURCES_DIR_PATH, FileNames.JSES), 'r') as f:
        script_item = f.read()

    @staticmethod
    def get_html_element_script():
        return Script.script_item

class HtmlControls():
    @staticmethod
    def get_html_element_base(*args):
        pass

    @staticmethod
    def get_html_element_script():
        pass

class Buttons(HtmlControls):
    script = """
    function show_element(this_button_id, graphic_group) {
        query = "[id^=" + graphic_group + "]";
        var graphic_groups = document.querySelectorAll("div" + query);
            for (i = 0; i < graphic_groups.length; i++){
              if (graphic_groups[i].id == this_button_id){
                  graphic_groups[i].style.display = "block";
              }
                else {
                  graphic_groups[i].style.display = "none";
                }
            }
    }
    """

    # Item then using as button_item.format(value)
    button_item = "<button onclick=\"show_element(\'{0}\', \'{2}\')\">{1}</button>"

    @staticmethod
    def get_html_element_base(element_ids, element_names, graphic_group_prefix):
        button_items = [
            Buttons.button_item.format(element_id, element_name, graphic_group_prefix)
            for element_id, element_name in zip(element_ids, element_names)
        ]
        return '\n'.join(button_items)


    @staticmethod
    def get_html_element_script():
        return Buttons.script

class Dropdown(HtmlControls):
    script = """
    function show_graphic_for_chosen_dropdown_options(metric_name) {
        query = "[id^=" + metric_name + "]";
            
            var dropdowns = document.querySelectorAll("select" + query);
            chosen_graphic_id = metric_name + "_" + dropdowns[0].value;
            for (i = 1; i < dropdowns.length; i++)
                chosen_graphic_id += "_" + dropdowns[i].value;
            
            var graphics = document.querySelectorAll("div" + query);
            for (i = 0; i < graphics.length; i++){
              if (graphics[i].id == chosen_graphic_id){
                  graphics[i].style.display = "block";
              }
                else {
                  graphics[i].style.display = "none";
                }
            }
  
    }

    """

    head = """<select name=\"figures\" id=\"{0}\" onchange=\"show_graphic_for_chosen_dropdown_options('{1}')\">"""

    # Item then using as body_item.format(value)
    body_item_selected = """<option selected value=\"{0}\">{1}</option>"""
    body_item = """<option value=\"{0}\">{1}</option>"""

    end = """</select>"""

    @staticmethod
    def get_html_element_base(element_values, dropdown_id, metric_name):
        body_items = \
        [Dropdown.body_item_selected.format(element_values[0], element_values[0])] + \
        [
            Dropdown.body_item.format(value, value)
            for value in element_values[1:]
        ]

        return '\n'.join([
            Dropdown.head.format(dropdown_id, metric_name),
            '\t' + '\n\t'.join(body_items),
            Dropdown.end
        ])

    @staticmethod
    def get_html_element_script():
        return Dropdown.script




