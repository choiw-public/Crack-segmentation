from tensorflow import unstack
from tensorflow import shape
import os
import re


def get_shape(tensor):
    _static_shape = tensor.get_shape().as_list()
    _dynamic_shape = unstack(shape(tensor))
    _dims = [s[1] if s[0] is None else s[0] for s in zip(_static_shape, _dynamic_shape)]
    return _dims


def sort_nicely(a_list):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    a_list.sort(key=alphanum_key)


def list_getter(dir_name, extension, must_include=None):
    file_list = []
    if dir_name:
        for path, subdirs, files in os.walk(dir_name):
            for name in files:
                if name.lower().endswith(tuple(extension)):
                    if must_include:
                        if must_include in name:
                            file_list.append(os.path.join(path, name))
                    else:
                        file_list.append(os.path.join(path, name))
        sort_nicely(file_list)
    return file_list
