import tensorflow as tf


def tnsr(_tensor_name=None, _exclude_name=None):
    if _tensor_name is not None:
        _tensors = [tensor.name.encode("ascii")
                    for tensor in
                    tf.get_default_graph().as_graph_def().node
                    if _tensor_name in tensor.name.encode("ascii")]
        if _exclude_name is not None:
            _tensors = [tensor for tensor in _tensors if _exclude_name not in tensor]
        for i in _tensors:
            print("%s: %s" % (i, tf.get_default_graph().get_tensor_by_name(i + ":0").get_shape().as_list()))

    else:
        _tensors = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
        if _exclude_name is not None:
            _tensors = [tensor for tensor in _tensors if _exclude_name not in tensor]
        for tensor in _tensors:
            print("%s: %s" % (tensor.name, tensor.get_shape()))


def tnsr_brief(_tensor_name=None):
    if _tensor_name is not None:
        _tensors = [tensor.name.encode("ascii")
                    for tensor in
                    tf.get_default_graph().as_graph_def().node
                    if _tensor_name in tensor.name.encode("ascii")]
        for i in _tensors:
            name_split = i.split("/")
            print("%s: %s" % (
                "/".join(name_split[-2:]).encode("ascii"),
                tf.get_default_graph().get_tensor_by_name(i + ":0").get_shape().as_list()))

    else:
        all_tensors = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
        for tensor in all_tensors:
            name_split = tensor.name.split("/")
            print("%s: %s" % ("/".join(name_split[-2:]).encode("ascii"), tensor.get_shape()))


def tnsr_ends(level=None):
    if level is None:
        level = 3
    _tensors = [tensor.name.encode("ascii")
                for tensor in
                tf.get_default_graph().as_graph_def().node]
    filtered_tensor_names = []
    for tensor in _tensors:
        name_split = tensor.split("/")
        if len(name_split) >= level:
            joined_name = "/".join(name_split[:level])
            if joined_name not in filtered_tensor_names:
                filtered_tensor_names.append(joined_name)
    for filtered_name in filtered_tensor_names:
        _tensor_names = [tensor.name.encode("ascii")
                         for tensor in
                         tf.get_default_graph().as_graph_def().node
                         if filtered_name in tensor.name.encode("ascii")]
        print("%s: %s" % (_tensor_names[-1],
                          tf.get_default_graph().get_tensor_by_name(_tensor_names[-1] + ":0").get_shape().as_list()))


def var(_var_name=None):
    if _var_name is not None:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=_var_name)
    else:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for i in variables:
        print(i)


def current_scope():
    print(tf.contrib.framework.get_name_scope())
