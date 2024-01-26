"""
Helper functions
"""
import json
import os
import traceback
from datetime import datetime

def load_from_json_file(path):
    with open(path, 'r') as file:
        json_str_input = file.read()
    return json.loads(json_str_input)

def save_json_str_as_file(filename, json_str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as text_file:
        text_file.write(json_str)


def save_dict_as_json(dictionary: dict, dir_path: str, filename: str, sort_keys=True):
    if dir_path is not None and len(dir_path) > 0:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = dir_path + "/" + filename
    json_str = json.dumps(dictionary, indent=4, sort_keys=sort_keys)
    save_json_str_as_file(filename, json_str)

def save_object_as_json(object, dir_path="save", file="object.json"):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    object_dict: dict = object.__dict__.copy()

    for key in object_dict:
        if not isinstance(object_dict[key], (float, int, str, list, dict, tuple)):
            object_dict[key] = None
    save_dict_as_json(object_dict, dir_path, file)


def log_time(dir_name, start_time, filename="time.json"):
    end_time = datetime.now()
    time_needed = end_time - start_time
    time_dict: dict = {}
    time_dict['start_time'] = start_time.strftime("%d_%m_%H_%M")
    time_dict['end_time'] = end_time.strftime("%d_%m_%H_%M")
    time_dict['time_needed'] = time_needed.total_seconds()
    try:
        save_dict_as_json(time_dict, dir_name, filename)
    except Exception as e:
        traceback.print_exc()
        print("ERROR: time_dict: " + str(e))

