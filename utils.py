"""
工具箱
:author: Qizhi Li
"""
import json
import pickle


def read_file(type_, path):
    """
    读取文件, 文件类型:
        1. pickle
        2. json
        3. txt
    :param type_: str
            文件类型
    :param path: str
            文件路径
    :return data: Object
            文件中的数据
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
    写入文件, 文件类型:
        1. pickle
        2. json
        3. txt
    :param type_: str
            文件类型
    :param path: str
            文件路径
    :param data: Object
            需要写入的文件
    """
    if type_ == 'pkl':
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif type_ == 'json':
        with open(path, 'w') as f:
            json.dump(data, f)
    elif type_ == 'txt_w':
        # 写入txt
        with open(path, 'w') as f:
            f.writelines(data)
