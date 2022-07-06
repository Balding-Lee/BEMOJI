"""
Utils
:author: Qizhi Li
"""
import json
import pickle


def read_file(type_, path):
    """
    Reading file, the types of file contains:
        1. pickle
        2. json
        3. txt
    :param type_: str
            'pkl', 'json', 'txt'
    :param path: str
    :return data: Object
    """
    if type_ == 'pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif type_ == 'json':
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        return None

    return data


def write_file(type_, path, data):
    """
    Writing file, the types of file contains:
        1. pickle
        2. json
        3. txt
    :param type_: str
            'pkl', 'json', 'txt'
    :param path: str
    :param data: Object
    """
    if type_ == 'pkl':
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif type_ == 'json':
        with open(path, 'w') as f:
            json.dump(data, f)
    elif type_ == 'txt_w':
        with open(path, 'w') as f:
            f.writelines(data)
