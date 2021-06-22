import json


def load_json(file_path):
    with open(file_path, 'r') as f:
        json_file = json.load(f)
    return json_file