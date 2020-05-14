from functions.project_fn.misc_utils import get_tensor_shape, add_to_collection, gen_gaussian_kernel_by_sigma
# from tensorflow.contrib.nccl.ops import gen_nccl_ops
# from tensorflow.contrib.framework import add_model_variable
from functions.project_fn.tf_debug_tool import *
from functions.project_fn.loss_functions import ssim_components
from scipy.stats import skewnorm

import tensorflow as tf
import numpy as np
import warnings


def return_as_is(input_tensor):
    return input_tensor


def elu(input_tensor):
    return tf.where(tf.greater(input_tensor, 0), input_tensor, tf.exp(input_tensor) - 1)


def relu1(input_tensor):
    return tf.minimum(tf.maximum(input_tensor, 0.0), 1.0)


def steeper_leaky_relu(input_tensor, upper=1.0, lower=0.0, multiplier=0.5):
    # todo: this is most likely trash.
    under_lower = tf.where(tf.less(input_tensor, lower), input_tensor * multiplier, tf.zeros_like(input_tensor))
    above_upper = tf.where(tf.greater(input_tensor, upper), input_tensor * multiplier, tf.zeros_like(input_tensor))
    activated = under_lower + above_upper
    activated = tf.where(tf.equal(activated, 0.0), input_tensor, activated)
    return activated


def swish(x):
    return x * tf.nn.sigmoid(x)


def hswish(x):
    return x * tf.nn.relu6(x + 3) / 6.0


def add_decov_reg_to_graph(tensor, name="decov_loss"):
    """Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf "Reducing
    Overfitting In Deep Networks by Decorrelating Representation".
    Args:
        tensor: 4-D `tensor` [batch_size, height, width, channels], input
    Returns:
        a `float` decov loss
    """
    with tf.name_scope(name):
        x = tf.reshape(tensor, [int(tensor.get_shape()[0]), -1])
        m = tf.reduce_mean(x, 0, True)
        z = tf.expand_dims(x - m, 2)
        corr = tf.reduce_mean(tf.matmul(z, tf.transpose(z, perm=[0, 2, 1])), 0)
        corr_frob_sqr = tf.reduce_sum(tf.square(corr))
        corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
        loss = 0.5 * (corr_frob_sqr - corr_diag_sqr)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)


def add_ssim_reg_to_graph(tensor, name="ssim_reg"):
    with tf.name_scope(name):
        _, _, _, c = get_tensor_shape(tensor)
        rnd_idx = tf.random.uniform([2], maxval=c, dtype=tf.int32)
        feature1 = tf.expand_dims(tensor[:, :, :, rnd_idx[0]], 3)
        feature2 = tf.expand_dims(tensor[:, :, :, rnd_idx[1]], 3)
        kernel = gen_gaussian_kernel_by_sigma(3, 1.5)
        minval = tf.reduce_min([feature1, feature2])
        feature1 -= minval
        feature2 -= minval
        _, _, lcs_term = ssim_components(feature1, feature2, kernel)
        ssim_reg = 0.5 * tf.reduce_mean(lcs_term) ** 2
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, ssim_reg)


def batch_instance_norm(x, scope="bin"):
    with tf.variable_scope(scope):
        ch = x.get_shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0), constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat


def get_fixup(name, fixup_type, dtype):
    if fixup_type == "mult":
        kernel_shape = [1]
        init = tf.initializers.ones()
    elif fixup_type == "bias":
        kernel_shape = [1]
        init = tf.initializers.zeros()
    else:
        raise ValueError("unexpected")
    return tf.get_variable(name, [1], dtype, init, None, True)


def fixup(tensor):
    kernel_shape = [1, 1, 1, 1]
    mult_init = tf.initializers.ones()
    fixup_mult = tf.get_variable("fixup_mult", kernel_shape, tensor.dtype, mult_init, None, True)
    bias_init = tf.initializers.zeros()
    fixup_bias = tf.get_variable("fixup_bias", kernel_shape, tensor.dtype, bias_init, None, True)
    return tensor * fixup_mult + fixup_bias


def fixup2(tensor):
    _, _, _, c = get_tensor_shape(tensor)
    mult_init = tf.initializers.ones()
    fixup_mult = tf.get_variable("fixup_mult", [c], tensor.dtype, mult_init, None, True)
    bias_init = tf.initializers.zeros()
    fixup_bias = tf.get_variable("fixup_bias", [c], tensor.dtype, bias_init, None, True)
    return tensor * fixup_mult + fixup_bias


def norm_and_actv(tensor, actv_fn, norm_type, norm_sequence, is_train, do_ssim_reg=False):
    def do_norm(_tensor):
        if norm_type == "fixup":
            _tensor = fixup(_tensor)
        elif norm_type == "fixup2":
            _tensor = fixup2(_tensor)
        elif norm_type == "bnorm":
            _tensor = tf.layers.batch_normalization(_tensor, training=is_train)
        elif norm_type == "bnorm_re":
            _tensor = tf.layers.batch_normalization(_tensor, renorm=True, training=is_train)
        elif norm_type is None:
            pass
        else:
            raise ValueError("unexpected norm_and_actv type")
        return _tensor

    if norm_sequence == "after_actv":
        tensor = actv_fn(tensor)
        tensor = do_norm(tensor)
        if do_ssim_reg and is_train:
            add_ssim_reg_to_graph(tensor)
    elif norm_sequence == "before_actv":
        tensor = do_norm(tensor)
        tensor = actv_fn(tensor)
        if do_ssim_reg and is_train:
            add_ssim_reg_to_graph(tensor)
    else:
        raise ValueError("unexpected norm_sequence")
    return tensor


def get_kernel(input_tensor, kernel_size, num_filters, weight_decay, kernel_type, init_factor=None, transpose=False):
    # important node: "num_filters" indicates channel multiplier if "kernel_type" is dw
    in_channel = get_tensor_shape(input_tensor)[-1]
    if transpose:
        kernel_shape = [kernel_size, kernel_size, num_filters, in_channel]
    else:
        kernel_shape = [kernel_size, kernel_size, in_channel, num_filters]
    if kernel_type == "he_uniform":
        init = tf.initializers.he_uniform()
    elif kernel_type == "dw":
        in_node = tf.cast(tf.reduce_prod(kernel_shape[0:2]), input_tensor.dtype)
        limit = tf.sqrt(6.0 / (in_node * init_factor))
        init = tf.initializers.random_uniform(minval=-1.0 * limit, maxval=limit)
    elif kernel_type in ["custom", "xavier_uniform"]:
        # init = tf.contrib.layers.xavier_initializer()
        # if weight_decay:
        #     regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
        # else:
        #     regularizer = None
        # return tf.get_variable("kernel", kernel_shape, tf.float32, init, regularizer, True)
        in_node = tf.cast(tf.reduce_prod(kernel_shape[0:3]), input_tensor.dtype)
        limit = tf.sqrt(6.0 / (in_node * init_factor))
        init = tf.initializers.random_uniform(minval=-1.0 * limit, maxval=limit)
    else:
        raise ValueError
    if weight_decay:
        regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    else:
        regularizer = None
    return tf.get_variable("kernel", kernel_shape, input_tensor.dtype, init, regularizer, True)


def get_activation_fn(actv_name):
    if actv_name == "relu":
        return tf.nn.relu
    elif actv_name == "elu":
        # return elu
        return tf.nn.elu
    elif actv_name == "selu":
        return tf.nn.selu
    elif actv_name == "sigmoid":
        return tf.sigmoid
    elif actv_name == "softmax":
        return tf.nn.softmax
    elif actv_name is None:
        return return_as_is
    elif actv_name == "relu1":
        return relu1
    elif actv_name == "swish":
        return swish
    elif actv_name == "hswish":
        return hswish
    elif actv_name == "steeper_leaky_relu":
        return steeper_leaky_relu
    else:
        raise ValueError("unsupported activation function")


def reshape(input_tensor, dims_list):
    shape = get_tensor_shape(input_tensor)
    dims_prod = []
    for dims in dims_list:
        if isinstance(dims, int):
            dims_prod.append(shape[dims])
        elif all([isinstance(shape[d], int) for d in dims]):
            dims_prod.append(np.prod([shape[d] for d in dims]))
        else:
            dims_prod.append(np.prod([shape[d] for d in dims]))
    input_tensor = tf.reshape(input_tensor, dims_prod)
    return input_tensor


def get_tensor_by_name(tensor_name):
    tensor_list = [tensor.name.encode("ascii")
                   for tensor in
                   tf.get_default_graph().as_graph_def().node
                   if tensor_name in tensor.name.encode("ascii")]
    if "BatchNorm" in tensor_list[-1]:
        tensor_name = [_name for _name in tensor_list if "batchnorm" in _name][-1] + ":0"
    else:
        tensor_name = tensor_list[-1] + ":0"
    if tensor_name:
        return tf.get_default_graph().get_tensor_by_name(tensor_name)
    else:
        raise ValueError("no tensor named by %s exists" % tensor_name)


def upscalesameas(tensor_to_be_upsampled, tensor_to_be_referenced):
    tensor_shape = tf.shape(tensor_to_be_referenced)
    height = tensor_shape[1]
    width = tensor_shape[2]
    return tf.cast(tf.image.resize_bilinear(tensor_to_be_upsampled,
                                            [height, width],
                                            align_corners=True,
                                            name="bilinear"), dtype=tensor_to_be_upsampled.dtype)


def pad(input_tensor, kernel_size, pad_type):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size should be an odd number")
    if pad_type == "VALID":
        return input_tensor
    else:
        pad = (kernel_size - 1) / 2
        return tf.pad(input_tensor, [[0, 0], [pad, pad], [pad, pad], [0, 0]], pad_type)


def add_bias(tensor):
    bias = tf.get_variable("bias", get_tensor_shape(tensor)[-1], dtype=tensor.dtype, initializer=tf.zeros_initializer())
    return tf.nn.bias_add(tensor, bias)


def depthwise_conv(input_tensor, dw_kernel, stride, padding="VALID", rate=1, use_bias=False):
    stride = [1, stride, stride, 1]
    out_tensor = tf.nn.depthwise_conv2d(input_tensor, dw_kernel, stride, padding.upper(), rate=(rate, rate))
    if use_bias:
        out_tensor = add_bias(out_tensor)
    return out_tensor


def conv(input_tensor, kernel, stride, use_bias=False, padding="SAME"):
    out_tensor = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding)
    if use_bias:
        out_tensor = add_bias(out_tensor)
    return out_tensor


def diminishing_bnorm_trial1(input_tensor, diminish):
    mean, variance = tf.nn.moments(input_tensor, 0)
    mean = mean * diminish
    variance = variance ** diminish
    return (input_tensor - mean) / tf.sqrt(variance + tf.constant(1e-8))


def diminishing_bnorm(input_tensor, diminish):
    mean, variance = tf.nn.moments(input_tensor, 0)
    mean = mean - (mean * diminish)
    std = tf.sqrt(variance)
    if input_tensor.dtype == tf.float16:
        epsilon = 1e-4
    elif input_tensor.dtype == tf.float32:
        epsilon = 1e-8
    else:
        raise ValueError("unexpected")
    std = std - (std - 1) * diminish + epsilon
    return (input_tensor - mean) / std


def stdconv_module(tensor_in,
                   out_depths,
                   pad_type,
                   weight_decay,
                   strides,
                   scope,
                   kernel_sizes,
                   use_conv_bias,
                   drop_rate,
                   is_train):
    actv_fn = get_activation_fn("elu")
    tensor_out = tensor_in
    with tf.variable_scope(scope):
        for i, depth in enumerate(out_depths):
            with tf.variable_scope("std_conv%s" % (i + 1)):
                kernel = get_kernel(tensor_out, kernel_sizes[i], depth, weight_decay, "conv")
                tensor_out = pad(tensor_out, kernel_sizes[i], pad_type)
                tensor_out = conv(tensor_out, kernel, strides[i], use_conv_bias, padding="VALID")
                tensor_out = actv_fn(tensor_out)
                if drop_rate != 0.0 and is_train:
                    tensor_out = tf.nn.dropout(tensor_out, 1 - drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {tensor_out.name.encode("ascii"): tf.math.reduce_variance(tensor_out)})
                if i == 0:
                    decode_feature = tensor_out
        return tensor_out, decode_feature


def stdconv_module2(tensor_in,
                    out_depths,
                    init_factors,
                    pad_type,
                    weight_decay,
                    strides,
                    scope,
                    kernel_sizes,
                    use_conv_bias,
                    drop_rate,
                    is_train,
                    actv="elu"):
    actv_fn = get_activation_fn(actv)
    tensor_out = tensor_in
    with tf.variable_scope(scope):
        for i, depth in enumerate(out_depths):
            with tf.variable_scope("std_conv%s" % (i + 1)):
                kernel = get_kernel(tensor_out, kernel_sizes[i], depth, weight_decay, "custom", init_factors[i])
                tensor_out = pad(tensor_out, kernel_sizes[i], pad_type)
                tensor_out = conv(tensor_out, kernel, strides[i], use_conv_bias, padding="VALID")
                tensor_out = actv_fn(tensor_out)
                if drop_rate != 0.0 and is_train:
                    tensor_out = tf.nn.dropout(tensor_out, rate=drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {tensor_out.name.encode("ascii"): tf.math.reduce_variance(tensor_out)})
                if i == 0:
                    decode_feature = tensor_out
        return tensor_out, decode_feature


def conv_block(tensor_in, kernel_size, stride, out_depth, weight_decay, is_train, bnorm_trainable, efficient):
    def build(main_pipe):
        kernel = get_kernel(main_pipe, kernel_size, out_depth, weight_decay, "he_uniform")
        main_pipe = tf.nn.conv2d(main_pipe, kernel, [1, stride, stride, 1], "SAME")
        main_pipe = tf.layers.batch_normalization(main_pipe, training=is_train, trainable=bnorm_trainable, fused=True)
        main_pipe = tf.nn.elu(main_pipe)
        return main_pipe

    if efficient:
        build = tf.contrib.layers.recompute_grad(build)
    return build(tensor_in)


def transpose_conv_block(tensor_in, kernel_size, stride, out_depth, out_shape, weight_decay, is_train, bnorm_trainable, efficient):
    def build(main_pipe):
        kernel = get_kernel(main_pipe, kernel_size, out_depth, weight_decay, "he_uniform", transpose=True)
        main_pipe = tf.nn.conv2d_transpose(main_pipe, kernel, out_shape, [1, stride, stride, 1], "SAME")
        main_pipe = tf.layers.batch_normalization(main_pipe, training=is_train, trainable=bnorm_trainable, fused=True)
        main_pipe = tf.nn.elu(main_pipe)
        return main_pipe

    if efficient:
        build = tf.contrib.layers.recompute_grad(build)
    return build(tensor_in)


def convolution(tensor_in, kernel_size, stride, out_depth, weight_decay, scope, is_train, bnorm_trainable, efficient):
    with tf.variable_scope(scope):
        return conv_block(tensor_in, kernel_size, stride, out_depth, weight_decay, is_train, bnorm_trainable, efficient)


def encoder_concat_shortcut(high_level, low_level, kernel_size, stride, out_depth, weight_decay, scope, is_train, bnorm_trainable, efficient):
    with tf.variable_scope(scope):
        low_level = conv_block(low_level, kernel_size, stride, out_depth, weight_decay, is_train, bnorm_trainable, efficient)
        return tf.concat([high_level, low_level], 3)


def upsample_transpose_conv(high_level, low_level, kernel_size, stride, out_depth, weight_decay, scope, is_train, bnorm_trainable, efficient):
    out_shape = get_tensor_shape(low_level)
    out_shape[-1] = out_depth
    with tf.variable_scope(scope):
        with tf.variable_scope("conv_transpose"):
            main_pipe = transpose_conv_block(high_level, kernel_size, stride, out_depth, out_shape, weight_decay, is_train, bnorm_trainable, efficient)
        with tf.variable_scope("shortcut_res"):
            main_pipe += conv_block(low_level, 1, 1, get_tensor_shape(main_pipe)[-1], weight_decay, is_train, bnorm_trainable, efficient)
    return main_pipe


def logit(tensor_in, kernel_size, stride, out_depth, weight_decay, efficient):
    def build(main_pipe):
        kernel = get_kernel(main_pipe, kernel_size, out_depth, weight_decay, "he_uniform")
        main_pipe = tf.nn.conv2d(main_pipe, kernel, [1, stride, stride, 1], "SAME")
        return main_pipe

    with tf.variable_scope("logit"):
        if efficient:
            build = tf.contrib.layers.recompute_grad(build)
    return build(tensor_in)


def stdconv_module_endnorm(tensor_in,
                           out_depth,
                           norm_type,
                           norm_sequence,
                           pad_type,
                           weight_decay,
                           strides,
                           scope,
                           kernel_size,
                           is_train,
                           drop_rate,
                           diminish,
                           conv_bias,
                           actv="elu",
                           is_diminishing=True):
    actv_fn = get_activation_fn(actv)
    tensor_out = tensor_in
    with tf.variable_scope(scope):
        kernel = get_kernel(tensor_out, kernel_size, out_depth, weight_decay, "he_uniform")
        tensor_out = pad(tensor_out, kernel_size, pad_type)
        tensor_out = conv(tensor_out, kernel, strides, conv_bias, padding="VALID")
        if is_diminishing:
            tensor_out = diminishing_bnorm(tensor_out, diminish)
        tensor_out = norm_and_actv(tensor_out, actv_fn, norm_type, norm_sequence, is_train)
        if drop_rate != 0.0 and is_train:
            tensor_out = tf.nn.dropout(tensor_out, rate=drop_rate)
        if not tf.executing_eagerly():
            tf.add_to_collection("monitor", {tensor_out.name.encode("ascii"): tf.math.reduce_variance(tensor_out)})
        return tensor_out


def stdconv_module_everynorm2(tensor_in,
                              out_depths,
                              init_factors,
                              norm_type,
                              norm_sequence,
                              pad_type,
                              weight_decay,
                              strides,
                              scope,
                              kernel_sizes,
                              is_train,
                              drop_rate,
                              actv="elu"):
    if norm_type is None:
        use_bias = True
    else:
        use_bias = False
    actv_fn = get_activation_fn(actv)
    tensor_out = tensor_in
    with tf.variable_scope(scope):
        for i, depth in enumerate(out_depths):
            with tf.variable_scope("std_conv%s" % (i + 1)):
                # kernel = get_kernel(tensor_out, kernel_size[i], depth, weight_decay, "conv")
                kernel = get_kernel(tensor_out, kernel_sizes[i], depth, weight_decay, "custom", init_factors[i])
                tensor_out = pad(tensor_out, kernel_sizes[i], pad_type)
                tensor_out = conv(tensor_out, kernel, strides[i], use_bias, padding="VALID")
                if i == len(out_depths) - 1:
                    tensor_out = norm_and_actv(tensor_out, actv_fn, norm_type, norm_sequence, is_train)
                if drop_rate != 0.0 and is_train:
                    tensor_out = tf.nn.dropout(tensor_out, rate=drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {tensor_out.name.encode("ascii"): tf.math.reduce_variance(tensor_out)})
                if i == 0:
                    decode_feature = tensor_out
        return tensor_out, decode_feature


def stdconv_module_everynorm2_he_normal(tensor_in,
                                        out_depths,
                                        norm_type,
                                        norm_sequence,
                                        pad_type,
                                        weight_decay,
                                        strides,
                                        scope,
                                        kernel_sizes,
                                        is_train,
                                        drop_rate,
                                        actv="elu"):
    if norm_type is None:
        use_bias = True
    else:
        use_bias = False
    actv_fn = get_activation_fn(actv)
    tensor_out = tensor_in
    with tf.variable_scope(scope):
        for i, depth in enumerate(out_depths):
            with tf.variable_scope("std_conv%s" % (i + 1)):
                kernel = get_kernel(tensor_out, kernel_sizes[i], depth, weight_decay, "he_uniform")
                tensor_out = pad(tensor_out, kernel_sizes[i], pad_type)
                tensor_out = conv(tensor_out, kernel, strides[i], use_bias, padding="VALID")
                if i == len(out_depths) - 1:
                    tensor_out = norm_and_actv(tensor_out, actv_fn, norm_type, norm_sequence, is_train)
                if drop_rate != 0.0 and is_train:
                    tensor_out = tf.nn.dropout(tensor_out, rate=drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {tensor_out.name.encode("ascii"): tf.math.reduce_variance(tensor_out)})
                if i == 0:
                    decode_feature = tensor_out
        return tensor_out, decode_feature


def densep_gc_module3(input_tensor,
                      pw_depths,
                      dw_sizes,
                      dw_multipliers,
                      dw_init_factors,
                      pad_type,
                      pw_weight_decay,
                      dw_weight_decay,
                      strides,
                      scope,
                      use_pw_bias,
                      use_dw_bias,
                      is_train,
                      pw_drop_rate,
                      dw_drop_rate,
                      gc_layer_norm=False,
                      gc_transform="multiplication"):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "pw")
                    main_pipe = conv(main_pipe, kernel, 1, use_pw_bias, "VALID")
                    main_pipe = actv_fn(main_pipe)
                    if pw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, 1 - pw_drop_rate)
                    # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=use_dw_bias)
                    if i <= len(dw_multipliers) - 2:
                        main_pipe = actv_fn(main_pipe)
                        if dw_drop_rate != 0.0 and is_train:
                            main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
                        if not tf.executing_eagerly():
                            tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        main_pipe = actv_fn(main_pipe)
        if dw_drop_rate != 0.0 and is_train:
            main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
        if not tf.executing_eagerly():
            tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep_gc_module3_2(input_tensor,
                        pw_depths,
                        pw_init_factors,
                        dw_sizes,
                        dw_multipliers,
                        dw_init_factors,
                        pad_type,
                        pw_weight_decay,
                        dw_weight_decay,
                        strides,
                        scope,
                        use_pw_bias,
                        use_dw_bias,
                        is_train,
                        pw_drop_rate,
                        dw_drop_rate,
                        gc_layer_norm=False,
                        gc_transform="multiplication"):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                    main_pipe = conv(main_pipe, kernel, 1, use_pw_bias, "VALID")
                    main_pipe = actv_fn(main_pipe)
                    if pw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, 1 - pw_drop_rate)
                    # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=use_dw_bias)
                    if i <= len(dw_multipliers) - 2:
                        main_pipe = actv_fn(main_pipe)
                        if dw_drop_rate != 0.0 and is_train:
                            main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
                        if not tf.executing_eagerly():
                            tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        main_pipe = actv_fn(main_pipe)
        if dw_drop_rate != 0.0 and is_train:
            main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
        if not tf.executing_eagerly():
            tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep(input_tensor,
           pw_depths,
           pw_init_factors,
           dw_sizes,
           dw_multipliers,
           dw_init_factors,
           fixup,
           pad_type,
           pw_weight_decay,
           dw_weight_decay,
           strides,
           scope,
           is_train,
           pw_drop_rate,
           dw_drop_rate):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                    if fixup:
                        main_pipe = main_pipe + get_fixup("pw_pre_bias", "bias", main_pipe.dtype)
                    main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                    if fixup:
                        main_pipe = main_pipe * get_fixup("pw_post_mult", "mult", main_pipe.dtype)
                        main_pipe = main_pipe + get_fixup("pw_post_bias", "bias", main_pipe.dtype)
                    main_pipe = actv_fn(main_pipe)
                    if pw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                    # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    if fixup:
                        main_pipe = main_pipe + get_fixup("dw_pre_bias", "bias", main_pipe.dtype)
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=True)
                    if fixup:
                        main_pipe = main_pipe * get_fixup("dw_post_mult", "mult", main_pipe.dtype)
                        main_pipe = main_pipe + get_fixup("dw_post_bias", "bias", main_pipe.dtype)
                    main_pipe = actv_fn(main_pipe)
                    if dw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep2(input_tensor,
            pw_depths,
            pw_init_factors,
            dw_sizes,
            dw_multipliers,
            dw_init_factors,
            do_fixup,
            pad_type,
            pw_weight_decay,
            dw_weight_decay,
            strides,
            scope,
            is_train,
            pw_drop_rate,
            dw_drop_rate,
            pw_actv="elu",
            dw_actv="elu"):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                    main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                    if do_fixup:
                        main_pipe = main_pipe * get_fixup("pw_post_mult", "mult", main_pipe.dtype)
                        main_pipe = main_pipe + get_fixup("pw_post_bias", "bias", main_pipe.dtype)
                    main_pipe = pw_actv_fn(main_pipe)
                    if pw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                    # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe

                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=True)
                    if do_fixup:
                        main_pipe = main_pipe * get_fixup("dw_post_mult", "mult", main_pipe.dtype)
                        main_pipe = main_pipe + get_fixup("dw_post_bias", "bias", main_pipe.dtype)
                    main_pipe = dw_actv_fn(main_pipe)
                    if dw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep2_everynorm(input_tensor,
                      pw_depths,
                      dw_sizes,
                      norm_type,
                      norm_sequence,
                      pad_type,
                      pw_weight_decay,
                      dw_weight_decay,
                      strides,
                      scope,
                      is_train,
                      pw_drop_rate,
                      dw_drop_rate,
                      diminish,
                      pw_bias,
                      dw_bias,
                      pw_actv="elu",
                      dw_actv="elu",
                      is_diminishing=True,
                      do_ssim_reg=False):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "he_uniform")
                    main_pipe = conv(main_pipe, kernel, 1, pw_bias, "VALID")
                    if is_diminishing:
                        main_pipe = diminishing_bnorm(main_pipe, diminish)
                    main_pipe = norm_and_actv(main_pipe, pw_actv_fn, norm_type, norm_sequence, is_train, do_ssim_reg)
                    if pw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], 1, dw_weight_decay, "he_uniform")
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=dw_bias)
                    if is_diminishing:
                        main_pipe = diminishing_bnorm(main_pipe, diminish)
                    main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                    if dw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def denconv_everynorm(input_tensor,
                      depths,
                      conv_size,
                      weight_decay,
                      strides,
                      scope,
                      is_train,
                      bnorm_trainable=True,
                      efficient=False,
                      do_gc=False,
                      gc_factor=None):
    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("denconv"):
            for i, depth in enumerate(depths):
                branches.append(main_pipe)
                with tf.variable_scope("invdenconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    main_pipe = conv_block(main_pipe, 1, 1, depth, weight_decay, is_train, bnorm_trainable, efficient)
                    if i == len(depths) - 1:
                        if do_gc:
                            main_pipe = gc_block(main_pipe, gc_factor, efficient, scope="gc_main")
                        decode_feature = main_pipe
                with tf.variable_scope("invdenconv%02d/conv" % (i + 1)):
                    main_pipe = conv_block(main_pipe, conv_size[i], strides[i], depth, weight_decay, is_train, bnorm_trainable, efficient)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def multiscale(input_tensor,
               depths,
               conv_sizes,
               out_depth,
               weight_decay,
               strides,
               scope,
               is_train,
               efficient,
               do_gc=False,
               gc_factor=None):
    with tf.variable_scope(scope):
        branches = []
        with tf.variable_scope("multiscale"):
            for i, depth in enumerate(depths):
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = conv_block(input_tensor, conv_sizes[i], 1, depth, weight_decay, is_train, efficient)
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = conv_block(main_pipe, conv_sizes[i], strides[i], depth, weight_decay, is_train, efficient)
                branches.append(main_pipe)
            main_pipe = tf.concat(branches, 3)
            main_pipe = conv_block(main_pipe, 1, 1, out_depth, weight_decay, is_train, efficient)
            if do_gc:
                main_pipe = gc_block(main_pipe, gc_factor, efficient)
        return main_pipe


def densep2_he_uniform(input_tensor,
                       pw_depths,
                       dw_sizes,
                       dw_multipliers,
                       pad_type,
                       pw_weight_decay,
                       dw_weight_decay,
                       strides,
                       scope,
                       is_train,
                       pw_drop_rate,
                       dw_drop_rate,
                       pw_actv="elu",
                       dw_actv="elu"):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "he_uniform")
                    main_pipe = conv(main_pipe, kernel, 1, False, "VALID")
                    main_pipe = norm_and_actv(main_pipe, pw_actv_fn, "bnorm", "before_actv", is_train)
                    if pw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe

                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "he_uniform")
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=False)
                    main_pipe = norm_and_actv(main_pipe, dw_actv_fn, "bnorm", "before_actv", is_train)
                    if dw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep_everynorm(input_tensor,
                     pw_depths,
                     pw_init_factors,
                     dw_sizes,
                     dw_multipliers,
                     dw_init_factors,
                     norm_type,
                     norm_sequence,
                     pad_type,
                     pw_weight_decay,
                     dw_weight_decay,
                     strides,
                     scope,
                     is_train,
                     pw_actv="elu",
                     dw_actv="elu"):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                    main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                    main_pipe = norm_and_actv(main_pipe, pw_actv_fn, norm_type, norm_sequence, is_train)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=True)
                    main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep_everynorm2(input_tensor,
                      pw_depths,
                      pw_init_factor,
                      dw_sizes,
                      dw_multipliers,
                      dw_init_factor,
                      out_depth,
                      norm_type,
                      norm_sequence,
                      pad_type,
                      pw_weight_decay,
                      dw_weight_decay,
                      stride,
                      scope,
                      is_train,
                      drop_rate,
                      pw_actv,
                      dw_actv):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    if norm_type is None:
        use_bias = True
    else:
        use_bias = False

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        concat_features = [main_pipe]
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factor)
                    main_pipe = conv(main_pipe, kernel, 1, use_bias, "VALID")
                    main_pipe = norm_and_actv(main_pipe, pw_actv_fn, norm_type, norm_sequence, is_train)
                    if drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factor)
                    main_pipe = depthwise_conv(main_pipe, kernel, 1, "VALID", use_bias=use_bias)
                    main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                    if drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i % 2 != 0:
                        concat_features.append(main_pipe)
                        main_pipe = tf.concat(concat_features, 3)
        with tf.variable_scope("out"):
            with tf.variable_scope("invsepconv/pw_conv"):
                kernel = get_kernel(main_pipe, 1, out_depth, pw_weight_decay, "custom", pw_init_factor)
                main_pipe = conv(main_pipe, kernel, 1, use_bias, "VALID")
                main_pipe = norm_and_actv(main_pipe, pw_actv_fn, norm_type, norm_sequence, is_train)
                if drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
            with tf.variable_scope("invsepconv/dw_conv"):
                main_pipe = pad(main_pipe, 5, pad_type)
                kernel = get_kernel(main_pipe, 5, 1, dw_weight_decay, "dw", dw_init_factor)
                main_pipe = depthwise_conv(main_pipe, kernel, stride, "VALID", use_bias=use_bias)
                main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                if drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
        return main_pipe, decode_feature


def densep_everynorm2_he_normal(input_tensor,
                                pw_depths,
                                dw_sizes,
                                dw_multipliers,
                                out_depth,
                                norm_type,
                                norm_sequence,
                                pad_type,
                                pw_weight_decay,
                                dw_weight_decay,
                                stride,
                                scope,
                                is_train,
                                drop_rate,
                                pw_actv,
                                dw_actv):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    if norm_type is None:
        use_bias = True
    else:
        use_bias = False

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        concat_features = [main_pipe]
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "he_uniform")
                    main_pipe = conv(main_pipe, kernel, 1, use_bias, "VALID")
                    main_pipe = norm_and_actv(main_pipe, pw_actv_fn, norm_type, norm_sequence, is_train)
                    if drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "he_uniform")
                    main_pipe = depthwise_conv(main_pipe, kernel, 1, "VALID", use_bias=use_bias)
                    main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                    if drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i % 2 != 0:
                        concat_features.append(main_pipe)
                        main_pipe = tf.concat(concat_features, 3)
        with tf.variable_scope("out"):
            with tf.variable_scope("invsepconv/pw_conv"):
                kernel = get_kernel(main_pipe, 1, out_depth, pw_weight_decay, "he_uniform")
                main_pipe = conv(main_pipe, kernel, 1, use_bias, "VALID")
                main_pipe = norm_and_actv(main_pipe, pw_actv_fn, norm_type, norm_sequence, is_train)
                if drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
            with tf.variable_scope("invsepconv/dw_conv"):
                main_pipe = pad(main_pipe, 5, pad_type)
                kernel = get_kernel(main_pipe, 5, 1, dw_weight_decay, "he_uniform")
                main_pipe = depthwise_conv(main_pipe, kernel, stride, "VALID", use_bias=use_bias)
                main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                if drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
        return main_pipe, decode_feature


def densep_endnorm(input_tensor,
                   pw_depths,
                   pw_init_factors,
                   dw_sizes,
                   dw_multipliers,
                   dw_init_factors,
                   norm_type,
                   norm_sequence,
                   pad_type,
                   pw_weight_decay,
                   dw_weight_decay,
                   strides,
                   scope,
                   is_train,
                   pw_actv="elu",
                   dw_actv="elu"):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                    main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                    main_pipe = pw_actv_fn(main_pipe)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=True)
                    if i != len(pw_depths) - 1:
                        main_pipe = dw_actv_fn(main_pipe)
                    else:
                        main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep_endnorm2(input_tensor,
                    pw_depths,
                    pw_init_factor,
                    dw_sizes,
                    dw_multipliers,
                    dw_init_factor,
                    out_depth,
                    norm_type,
                    norm_sequence,
                    pad_type,
                    pw_weight_decay,
                    dw_weight_decay,
                    stride,
                    scope,
                    is_train,
                    drop_rate,
                    pw_actv,
                    dw_actv):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        concat_features = [main_pipe]
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factor)
                    main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                    main_pipe = pw_actv_fn(main_pipe)
                    if drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factor)
                    main_pipe = depthwise_conv(main_pipe, kernel, 1, "VALID", use_bias=True)
                    main_pipe = dw_actv_fn(main_pipe)
                    if drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i % 2 != 0:
                        concat_features.append(main_pipe)
                        main_pipe = tf.concat(concat_features, 3)
        with tf.variable_scope("out"):
            with tf.variable_scope("invsepconv/pw_conv"):
                kernel = get_kernel(main_pipe, 1, out_depth, pw_weight_decay, "custom", pw_init_factor)
                main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                main_pipe = pw_actv_fn(main_pipe)
                if drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
            with tf.variable_scope("invsepconv/dw_conv"):
                main_pipe = pad(main_pipe, 5, pad_type)
                kernel = get_kernel(main_pipe, 5, 1, dw_weight_decay, "dw", dw_init_factor)
                main_pipe = depthwise_conv(main_pipe, kernel, stride, "VALID", use_bias=True)
                main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                if drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=drop_rate)
        return main_pipe, decode_feature


def sep(input_tensor,
        pw_depth,
        pw_init_factor,
        dw_size,
        dw_multiplier,
        dw_init_factor,
        fixup,
        pad_type,
        pw_weight_decay,
        dw_weight_decay,
        strides,
        scope,
        is_train,
        pw_drop_rate,
        dw_drop_rate):
    actv_fn = get_activation_fn("elu")
    main_pipe = input_tensor
    with tf.variable_scope(scope):
        with tf.variable_scope("sep"):
            with tf.variable_scope("pw_conv"):
                kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factor)
                if fixup:
                    main_pipe = main_pipe + get_fixup("pw_pre_bias", "bias", main_pipe.dtype)
                main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                if fixup:
                    main_pipe = main_pipe * get_fixup("pw_post_mult", "mult", main_pipe.dtype)
                    main_pipe = main_pipe + get_fixup("pw_post_bias", "bias", main_pipe.dtype)
                main_pipe = actv_fn(main_pipe)
                if pw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
            with tf.variable_scope("dw_conv"):
                main_pipe = pad(main_pipe, dw_size, pad_type)
                kernel = get_kernel(main_pipe, dw_size, dw_multiplier, dw_weight_decay, "dw", dw_init_factor)
                if fixup:
                    main_pipe = main_pipe + get_fixup("dw_pre_bias", "bias", main_pipe.dtype)
                main_pipe = depthwise_conv(main_pipe, kernel, strides, "VALID", use_bias=True)
                if fixup:
                    main_pipe = main_pipe * get_fixup("dw_post_mult", "mult", main_pipe.dtype)
                    main_pipe = main_pipe + get_fixup("dw_post_bias", "bias", main_pipe.dtype)
                main_pipe = actv_fn(main_pipe)
                if dw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe


def sep2(input_tensor,
         pw_depth,
         pw_init_factor,
         dw_size,
         dw_multiplier,
         dw_init_factor,
         fixup,
         pad_type,
         pw_weight_decay,
         dw_weight_decay,
         strides,
         scope,
         is_train,
         pw_drop_rate,
         dw_drop_rate):
    actv_fn = get_activation_fn("elu")
    main_pipe = input_tensor
    with tf.variable_scope(scope):
        with tf.variable_scope("sep"):
            with tf.variable_scope("pw_conv"):
                kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factor)
                main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                if fixup:
                    main_pipe = main_pipe * get_fixup("pw_post_mult", "mult", main_pipe.dtype)
                    main_pipe = main_pipe + get_fixup("pw_post_bias", "bias", main_pipe.dtype)
                main_pipe = actv_fn(main_pipe)
                if pw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
            with tf.variable_scope("dw_conv"):
                main_pipe = pad(main_pipe, dw_size, pad_type)
                kernel = get_kernel(main_pipe, dw_size, dw_multiplier, dw_weight_decay, "dw", dw_init_factor)
                main_pipe = depthwise_conv(main_pipe, kernel, strides, "VALID", use_bias=True)
                if fixup:
                    main_pipe = main_pipe * get_fixup("dw_post_mult", "mult", main_pipe.dtype)
                    main_pipe = main_pipe + get_fixup("dw_post_bias", "bias", main_pipe.dtype)
                main_pipe = actv_fn(main_pipe)
                if dw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe


def sep2_everynorm_nonoverlapping(input_tensor,
                                  pw_depth,
                                  dw_size,
                                  norm_type,
                                  norm_sequence,
                                  pw_weight_decay,
                                  dw_weight_decay,
                                  scope,
                                  is_train,
                                  pw_drop_rate,
                                  dw_drop_rate,
                                  diminish,
                                  pw_bias,
                                  dw_bias,
                                  pw_actv="elu",
                                  dw_actv="elu",
                                  is_diminishing=True,
                                  do_decov=False):
    pw_actv_fn = get_activation_fn(pw_actv)
    dw_actv_fn = get_activation_fn(dw_actv)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        with tf.variable_scope("sep"):
            with tf.variable_scope("pw_conv"):
                kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "he_uniform")
                main_pipe = conv(main_pipe, kernel, 1, pw_bias, "VALID")
                if is_diminishing:
                    main_pipe = diminishing_bnorm(main_pipe, diminish)
                main_pipe = norm_and_actv(main_pipe, pw_actv_fn, norm_type, norm_sequence, is_train, do_decov)
                if pw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
            with tf.variable_scope("dw_conv"):
                kernel = get_kernel(main_pipe, dw_size, 1, dw_weight_decay, "he_uniform")
                main_pipe = depthwise_conv(main_pipe, kernel, 1, "VALID", use_bias=dw_bias)
                if is_diminishing:
                    main_pipe = diminishing_bnorm(main_pipe, diminish)
                main_pipe = norm_and_actv(main_pipe, dw_actv_fn, norm_type, norm_sequence, is_train)
                if dw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe


def pw_pooling(input_tensor,
               pw_depth,
               pw_init_factor,
               fixup,
               pw_weight_decay,
               scope,
               is_train,
               pw_drop_rate):
    actv_fn = get_activation_fn("elu")
    main_pipe = input_tensor
    with tf.variable_scope(scope):
        with tf.variable_scope("pw_conv"):
            kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factor)
            if fixup:
                main_pipe = main_pipe + get_fixup("pw_pre_bias", "bias", main_pipe.dtype)
            main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
            if fixup:
                main_pipe = main_pipe * get_fixup("pw_post_mult", "mult", main_pipe.dtype)
                main_pipe = main_pipe + get_fixup("pw_post_bias", "bias", main_pipe.dtype)
            main_pipe = actv_fn(main_pipe)
            if pw_drop_rate != 0.0 and is_train:
                main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
        with tf.variable_scope("avg_pooling"):
            _, h, w, _ = get_tensor_shape(main_pipe)
            main_pipe = tf.nn.avg_pool(main_pipe, [1, h, w, 1], [1, 1, 1, 1], "VALID")
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe


def sep_global(input_tensor,
               pw_depth,
               pw_init_factor,
               dw_init_factor,
               fixup,
               pw_weight_decay,
               dw_weight_decay,
               scope,
               is_train,
               pw_drop_rate,
               dw_drop_rate):
    actv_fn = get_activation_fn("elu")
    main_pipe = input_tensor
    with tf.variable_scope(scope):
        with tf.variable_scope("pw_conv"):
            kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factor)
            if fixup:
                main_pipe = main_pipe + get_fixup("pw_pre_bias", "bias", main_pipe.dtype)
            main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
            if fixup:
                main_pipe = main_pipe * get_fixup("pw_post_mult", "mult", main_pipe.dtype)
                main_pipe = main_pipe + get_fixup("pw_post_bias", "bias", main_pipe.dtype)
            main_pipe = actv_fn(main_pipe)
            if pw_drop_rate != 0.0 and is_train:
                main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
        with tf.variable_scope("global_dw"):
            _, h, w, _ = get_tensor_shape(main_pipe)
            if h != w:
                raise ValueError("h should be equal to w")
            kernel = get_kernel(main_pipe, h, 1, dw_weight_decay, "dw", dw_init_factor)
            if fixup:
                main_pipe = main_pipe + get_fixup("dw_pre_bias", "bias", main_pipe.dtype)
            main_pipe = depthwise_conv(main_pipe, kernel, 1, "VALID", use_bias=True)
            if fixup:
                main_pipe = main_pipe * get_fixup("dw_post_mult", "mult", main_pipe.dtype)
                main_pipe = main_pipe + get_fixup("dw_post_bias", "bias", main_pipe.dtype)
            main_pipe = actv_fn(main_pipe)
            if dw_drop_rate != 0.0 and is_train:
                main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe


def pw(input_tensor,
       pw_depth,
       pw_weight_decay,
       scope,
       is_train,
       pw_drop_rate,
       stride=1,
       actv="elu",
       do_decov=False):
    actv_fn = get_activation_fn(actv)
    main_pipe = input_tensor
    with tf.variable_scope(scope):
        kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "he_uniform")
        main_pipe = conv(main_pipe, kernel, stride, True, "VALID")
        main_pipe = actv_fn(main_pipe)
        if do_decov and is_train:
            add_decov_reg_to_graph(main_pipe)
        if pw_drop_rate != 0.0 and is_train:
            main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
    return main_pipe


def pw_he_normal(input_tensor,
                 pw_depth,
                 norm_type,
                 norm_sequence,
                 pw_weight_decay,
                 scope,
                 is_train,
                 pw_drop_rate,
                 stride=1,
                 actv="elu"):
    actv_fn = get_activation_fn(actv)
    main_pipe = input_tensor
    with tf.variable_scope(scope):
        kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "he_uniform")
        main_pipe = conv(main_pipe, kernel, stride, True, "VALID")
        main_pipe = norm_and_actv(main_pipe, actv_fn, norm_type, norm_sequence, is_train)
        if pw_drop_rate != 0.0 and is_train:
            main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
    return main_pipe


def densep_fixup_tmp(input_tensor,
                     pw_depths,
                     pw_init_factors,
                     dw_sizes,
                     dw_multipliers,
                     dw_init_factors,
                     pad_type,
                     pw_weight_decay,
                     dw_weight_decay,
                     strides,
                     scope,
                     is_train,
                     pw_drop_rate,
                     dw_drop_rate):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                    main_pipe = main_pipe + get_fixup("pw_pre_bias", "bias", main_pipe.dtype)
                    main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                    main_pipe = main_pipe * get_fixup("pw_post_mult", "mult", main_pipe.dtype)
                    main_pipe = main_pipe + get_fixup("pw_post_bias", "bias", main_pipe.dtype)
                    main_pipe = actv_fn(main_pipe)
                    if pw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, 1 - pw_drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    main_pipe = main_pipe + get_fixup("dw_pre_bias", "bias", main_pipe.dtype)
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=True)
                    main_pipe = main_pipe * get_fixup("dw_post_mult", "mult", main_pipe.dtype)
                    main_pipe = main_pipe + get_fixup("dw_post_bias", "bias", main_pipe.dtype)
                    main_pipe = actv_fn(main_pipe)
                    if dw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep_transition_fixup(input_tensor,
                            pw_depths,
                            pw_init_factors,
                            dw_sizes,
                            dw_multipliers,
                            dw_init_factors,
                            pad_type,
                            pw_weight_decay,
                            dw_weight_decay,
                            strides,
                            scope,
                            is_train,
                            pw_drop_rate,
                            dw_drop_rate):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        for i, pw_depth in enumerate(pw_depths):
            branches.append(main_pipe)
            if i == len(pw_depths) - 1 and len(pw_depths) > 1:
                main_pipe = tf.concat(branches[1::], 3)
                subscope = "transition"
            else:
                main_pipe = tf.concat(branches, 3)
                subscope = "invsepconv%02d" % (i + 1)
            with tf.variable_scope(subscope + "/pw_conv"):
                kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                main_pipe = main_pipe + get_fixup("pw_pre_bias", "bias", main_pipe.dtype)
                main_pipe = conv(main_pipe, kernel, 1, False, "VALID")
                main_pipe = main_pipe * get_fixup("pw_post_mult", "mult", main_pipe.dtype)
                main_pipe = main_pipe + get_fixup("pw_post_bias", "bias", main_pipe.dtype)
                main_pipe = actv_fn(main_pipe)
                if pw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                if i == len(pw_depths) - 1:
                    decode_feature = main_pipe
            with tf.variable_scope(subscope + "/dw_conv"):
                main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                main_pipe = main_pipe + get_fixup("dw_pre_bias", "bias", main_pipe.dtype)
                main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=False)
                main_pipe = main_pipe * get_fixup("dw_post_mult", "mult", main_pipe.dtype)
                main_pipe = main_pipe + get_fixup("dw_post_bias", "bias", main_pipe.dtype)
                main_pipe = actv_fn(main_pipe)
                if dw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep_transition_fixup2(input_tensor,
                             pw_depths,
                             pw_init_factors,
                             dw_sizes,
                             dw_multipliers,
                             dw_init_factors,
                             pad_type,
                             pw_weight_decay,
                             dw_weight_decay,
                             strides,
                             scope,
                             is_train,
                             pw_drop_rate,
                             dw_drop_rate):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        for i, pw_depth in enumerate(pw_depths):
            branches.append(main_pipe)
            if i == len(pw_depths) - 1 and len(pw_depths) > 1:
                main_pipe = tf.concat(branches[1::], 3)
                subscope = "transition"
            else:
                main_pipe = tf.concat(branches, 3)
                subscope = "invsepconv%02d" % (i + 1)
            with tf.variable_scope(subscope + "/pw_conv"):
                kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                main_pipe = conv(main_pipe, kernel, 1, True, "VALID")
                main_pipe = main_pipe * tf.Variable(1.0, trainable=True, name="pw_post_mult", dtype=main_pipe.dtype)
                main_pipe = main_pipe + tf.Variable(0.0, trainable=True, name="pw_post_bias", dtype=main_pipe.dtype)
                main_pipe = actv_fn(main_pipe)
                if pw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=pw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                if i == len(pw_depths) - 1:
                    decode_feature = main_pipe
            with tf.variable_scope(subscope + "/dw_conv"):
                main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=True)
                main_pipe = main_pipe * tf.Variable(1.0, trainable=True, name="dw_post_mult", dtype=main_pipe.dtype)
                main_pipe = main_pipe + tf.Variable(0.0, trainable=True, name="dw_post_bias", dtype=main_pipe.dtype)
                main_pipe = actv_fn(main_pipe)
                if dw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, rate=dw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def middle_res(input_tensor,
               pw_depth,
               pw_squeeze_init_factor,
               dw_init_factor,
               pw_expansion_init_factor,
               pad_type,
               pw_weight_decay,
               dw_weight_decay,
               scope,
               use_pw_bias,
               use_dw_bias,
               is_train,
               pw_drop_rate,
               dw_drop_rate,
               gc_layer_norm=False,
               gc_transform="multiplication"):
    actv_fn = get_activation_fn("elu")
    _, _, _, c = get_tensor_shape(input_tensor)
    with tf.variable_scope(scope):
        with tf.variable_scope("pw_squeeze"):
            kernel = get_kernel(input_tensor, 1, pw_depth, pw_weight_decay, "custom", pw_squeeze_init_factor)
            main_pipe = conv(input_tensor, kernel, 1, use_pw_bias, "VALID")
            main_pipe = actv_fn(main_pipe)
            if pw_drop_rate != 0.0 and is_train:
                main_pipe = tf.nn.dropout(main_pipe, 1 - pw_drop_rate)
            # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        with tf.variable_scope("dw_conv"):
            main_pipe = pad(main_pipe, 3, pad_type)
            kernel = get_kernel(main_pipe, 3, 1, dw_weight_decay, "dw", dw_init_factor)
            main_pipe = depthwise_conv(main_pipe, kernel, 1, "VALID", use_bias=use_dw_bias)
            main_pipe = actv_fn(main_pipe)
            if dw_drop_rate != 0.0 and is_train:
                main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        with tf.variable_scope("pw_expansion"):
            kernel = get_kernel(main_pipe, 1, c, pw_weight_decay, "custom", pw_expansion_init_factor)
            main_pipe = conv(main_pipe, kernel, 1, use_pw_bias, "VALID")
            main_pipe = actv_fn(main_pipe)
            if pw_drop_rate != 0.0 and is_train:
                main_pipe = tf.nn.dropout(main_pipe, 1 - pw_drop_rate)
            # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return input_tensor + main_pipe


def densep_gc_module4(input_tensor,
                      pw_depths,
                      pw_init_factors,
                      dw_sizes,
                      dw_multipliers,
                      dw_init_factors,
                      pad_type,
                      pw_weight_decay,
                      dw_weight_decay,
                      strides,
                      scope,
                      use_pw_bias,
                      use_dw_bias,
                      is_train,
                      pw_drop_rate,
                      dw_drop_rate,
                      gc_layer_norm=False,
                      gc_transform="multiplication"):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)
    main_pipe = input_tensor
    densep_features = []
    with tf.variable_scope(scope):
        for i, pw_depth in enumerate(pw_depths):
            if i == 0:
                sub_scope = "representation"
            elif i == len(pw_depths) - 1:
                sub_scope = "transition"
            else:
                sub_scope = "densep" + str(i)
            with tf.variable_scope(sub_scope):
                kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                main_pipe = conv(main_pipe, kernel, 1, use_pw_bias, "VALID")
                main_pipe = actv_fn(main_pipe)
                if pw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, 1 - pw_drop_rate)
                # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                if i == len(pw_depths) - 1:
                    decode_feature = main_pipe
            with tf.variable_scope(sub_scope):
                main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=use_dw_bias)
                main_pipe = actv_fn(main_pipe)
                if dw_drop_rate != 0.0 and is_train:
                    main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
            if sub_scope != "transition":
                densep_features.append(main_pipe)
                main_pipe = tf.concat(densep_features, 3)
    return main_pipe, decode_feature


def densep_gc_module3_3(input_tensor,
                        pw_depths,
                        pw_init_factors,
                        dw_sizes,
                        dw_multipliers,
                        dw_init_factors,
                        pad_type,
                        pw_weight_decay,
                        dw_weight_decay,
                        strides,
                        scope,
                        use_pw_bias,
                        use_dw_bias,
                        is_train,
                        pw_drop_rate,
                        dw_drop_rate,
                        gc_layer_norm=False,
                        gc_transform="multiplication"):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "custom", pw_init_factors[i])
                    main_pipe = conv(main_pipe, kernel, 1, use_pw_bias, "VALID")
                    main_pipe = actv_fn(main_pipe)
                    if pw_drop_rate != 0.0 and is_train:
                        main_pipe = tf.nn.dropout(main_pipe, 1 - pw_drop_rate)
                    # main_pipe = gc_block(main_pipe, ratio=4, use_bias=True)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                    return main_pipe, None
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=use_dw_bias)
                    if i <= len(dw_multipliers) - 2:
                        main_pipe = actv_fn(main_pipe)
                        if dw_drop_rate != 0.0 and is_train:
                            main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
                        if not tf.executing_eagerly():
                            tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})

        main_pipe = actv_fn(main_pipe)
        if dw_drop_rate != 0.0 and is_train:
            main_pipe = tf.nn.dropout(main_pipe, 1 - dw_drop_rate)
        if not tf.executing_eagerly():
            tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def densep_gc_module3_bin(input_tensor,
                          pw_depths,
                          dw_sizes,
                          dw_multipliers,
                          dw_init_factors,
                          pad_type,
                          pw_weight_decay,
                          dw_weight_decay,
                          strides,
                          scope,
                          gc_layer_norm=False,
                          gc_transform="multiplication"):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)

    with tf.variable_scope(scope):
        main_pipe = input_tensor
        branches = []
        with tf.variable_scope("densep_module"):
            for i, pw_depth in enumerate(pw_depths):
                branches.append(main_pipe)
                with tf.variable_scope("invsepconv%02d/pw_conv" % (i + 1)):
                    main_pipe = tf.concat(branches, 3)
                    kernel = get_kernel(main_pipe, 1, pw_depth, pw_weight_decay, "pw")
                    main_pipe = conv(main_pipe, kernel, 1, False, "VALID")
                    main_pipe = batch_instance_norm(main_pipe)
                    main_pipe = actv_fn(main_pipe)
                    # main_pipe = global_context_block2(main_pipe, use_bias=True, layer_norm=gc_layer_norm, transform=gc_transform)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
                    if i == len(pw_depths) - 1:
                        decode_feature = main_pipe
                with tf.variable_scope("invsepconv%02d/dw_conv" % (i + 1)):
                    main_pipe = pad(main_pipe, dw_sizes[i], pad_type)
                    kernel = get_kernel(main_pipe, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                    main_pipe = depthwise_conv(main_pipe, kernel, strides[i], "VALID", use_bias=False)
                    if i <= len(dw_multipliers) - 2:
                        main_pipe = batch_instance_norm(main_pipe)
                        main_pipe = actv_fn(main_pipe)
                    if not tf.executing_eagerly():
                        tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        main_pipe = batch_instance_norm(main_pipe)
        main_pipe = actv_fn(main_pipe)
        if not tf.executing_eagerly():
            tf.add_to_collection("monitor", {main_pipe.name.encode("ascii"): tf.math.reduce_variance(main_pipe)})
        return main_pipe, decode_feature


def sep_gc_decoder3(high_level_feature,
                    low_level_feature,
                    pw_depths,
                    dw_sizes,
                    dw_multipliers,
                    dw_init_factors,
                    pad_type,
                    pw_weight_decay,
                    dw_weight_decay,
                    scope,
                    use_pw_bias=False,
                    use_dw_bias=False,
                    gc_layer_norm=False,
                    gc_transform="multiplication"):
    actv_fn = get_activation_fn("elu")

    if not dw_multipliers:
        dw_multipliers = [1] * len(pw_depths)
    branches = []
    with tf.variable_scope(scope):
        with tf.variable_scope("image_level_feature"):
            with tf.variable_scope("pw_conv"):
                _, _, _, in_channel = get_tensor_shape(low_level_feature)
                kernel = get_kernel(low_level_feature, 1, in_channel / 4, pw_weight_decay, "pw")
                low_level_feature = conv(low_level_feature, kernel, 1, use_pw_bias, "VALID")
                low_level_feature = actv_fn(low_level_feature)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {low_level_feature.name.encode("ascii"): tf.math.reduce_variance(low_level_feature)})
            with tf.variable_scope("dw_conv"):
                low_level_feature = pad(low_level_feature, 3, pad_type)
                kernel = get_kernel(low_level_feature, 3, 2, dw_weight_decay, "dw", 1.4)
                low_level_feature = depthwise_conv(low_level_feature, kernel, 1, "VALID", use_bias=use_dw_bias)
                low_level_feature = actv_fn(low_level_feature)
                if not tf.executing_eagerly():
                    tf.add_to_collection("monitor", {low_level_feature.name.encode("ascii"): tf.math.reduce_variance(low_level_feature)})
        with tf.variable_scope("decoding"):
            branches.append(upscalesameas(high_level_feature, low_level_feature))
            branches.append(low_level_feature)
            outputs = tf.concat(branches, 3)
            with tf.variable_scope("sep_decoder"):
                for i, pw_depth in enumerate(pw_depths):
                    with tf.variable_scope("invsepconv%02d" % (i + 1)):
                        with tf.variable_scope("pw_conv"):
                            kernel = get_kernel(outputs, 1, pw_depth, pw_weight_decay, "pw")
                            outputs = conv(outputs, kernel, 1, use_pw_bias, "VALID")
                            outputs = actv_fn(outputs)
                            # outputs = global_context_block2(outputs, use_bias=use_pw_bias, layer_norm=gc_layer_norm, transform=gc_transform)
                            if not tf.executing_eagerly():
                                tf.add_to_collection("monitor", {outputs.name.encode("ascii"): tf.math.reduce_variance(outputs)})
                            # add_to_collection(outputs)
                        with tf.variable_scope("dw_conv"):
                            outputs = pad(outputs, 3, pad_type)
                            kernel = get_kernel(outputs, dw_sizes[i], dw_multipliers[i], dw_weight_decay, "dw", dw_init_factors[i])
                            outputs = depthwise_conv(outputs, kernel, 1, "VALID", use_bias=use_dw_bias)
                            if i <= len(dw_multipliers) - 2:
                                outputs = actv_fn(outputs)
                            if not tf.executing_eagerly():
                                tf.add_to_collection("monitor", {outputs.name.encode("ascii"): tf.math.reduce_variance(outputs)})
            outputs = actv_fn(outputs)
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {outputs.name.encode("ascii"): tf.math.reduce_variance(outputs)})
            return outputs


def stdconv_decoder(input_tensor,
                    out_depth,
                    pad_type,
                    weight_decay,
                    scope,
                    kernel_size=3,
                    use_bias=True):
    with tf.variable_scope(scope):
        with tf.variable_scope("stdconv_decoder"):
            kernel = get_kernel(input_tensor, kernel_size, out_depth, weight_decay, "he_uniform")
            out_tensor = pad(input_tensor, kernel_size, pad_type)
            out_tensor = conv(out_tensor, kernel, 1, use_bias, "VALID")
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {out_tensor.name.encode("ascii"): tf.math.reduce_variance(out_tensor)})
        return out_tensor


def classification_end(input_tensor,
                       out_depth,
                       init_factor,
                       weight_decay,
                       drop_rate,
                       scope,
                       use_bias,
                       is_train):
    with tf.variable_scope(scope):
        with tf.variable_scope("avg_pooling"):
            _, h, w, _ = get_tensor_shape(input_tensor)
            out_tensor = tf.nn.avg_pool(input_tensor, [1, h, w, 1], [1, 1, 1, 1], "VALID")
            if drop_rate != 0.0 and is_train:
                out_tensor = tf.nn.dropout(out_tensor, rate=drop_rate)
        with tf.variable_scope("std_conv"):
            kernel = get_kernel(out_tensor, 1, out_depth, weight_decay, "custom", init_factor)
            out_tensor = conv(out_tensor, kernel, 1, use_bias, "VALID")
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {out_tensor.name.encode("ascii"): tf.math.reduce_variance(out_tensor)})
        return tf.squeeze(out_tensor)


def global_average_pooling(input_tensor):
    with tf.variable_scope("avg_pooling"):
        _, h, w, _ = get_tensor_shape(input_tensor)
        out_tensor = tf.nn.avg_pool(input_tensor, [1, h, w, 1], [1, 1, 1, 1], "VALID")
        return out_tensor


def gc_block(tensor_in, factor, efficient, scope="gc_block"):
    # GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
    def build(main_pipe):
        with tf.variable_scope(scope):
            with tf.variable_scope("context"):
                n, h, w, c = get_tensor_shape(main_pipe)
                tensor_in_flatten = tf.reshape(main_pipe, [n, h * w, c])
                kernel = get_kernel(main_pipe, 1, 1, None, "he_uniform")
                context = tf.nn.conv2d(main_pipe, kernel, strides=[1, 1, 1, 1], padding="SAME")
                context = tf.reshape(context, [n, h * w, 1])
                context = tf.nn.softmax(context, axis=1)
                context = tf.matmul(tensor_in_flatten, context, transpose_a=True)
                context = tf.reshape(context, [n, 1, 1, c])

            with tf.variable_scope("transform"):
                with tf.variable_scope("shrink"):
                    kernel = get_kernel(context, 1, int(c / factor), None, "he_uniform")
                    transform = tf.nn.conv2d(context, kernel, [1, 1, 1, 1], "SAME")
                    transform = tf.contrib.layers.layer_norm(transform, center=True, scale=True, scope=scope)
                    transform = tf.nn.relu(transform)
                with tf.variable_scope("expand"):
                    kernel = get_kernel(transform, 1, c, None, "he_uniform")
                    transform = tf.nn.conv2d(transform, kernel, [1, 1, 1, 1], "SAME")
                    transform = tf.nn.sigmoid(transform)
            return main_pipe + transform

    if efficient:
        build = tf.contrib.layers.recompute_grad(build)
    return build(tensor_in)


def classification_end_he_uniform(input_tensor,
                                  out_depth,
                                  weight_decay,
                                  drop_rate,
                                  scope,
                                  use_bias,
                                  do_decov,
                                  is_train):
    with tf.variable_scope(scope):
        with tf.variable_scope("std_conv"):
            kernel = get_kernel(input_tensor, 1, out_depth, weight_decay, "he_uniform")
            out_tensor = conv(input_tensor, kernel, 1, use_bias, "VALID")
            if do_decov and is_train:
                add_decov_reg_to_graph(out_tensor)
            if drop_rate != 0.0 and is_train:
                out_tensor = tf.nn.dropout(out_tensor, rate=drop_rate)
            if not tf.executing_eagerly():
                tf.add_to_collection("monitor", {out_tensor.name.encode("ascii"): tf.math.reduce_variance(out_tensor)})
        return tf.squeeze(out_tensor)


def classification_end2(input_tensor,
                        out_depth,
                        init_factor,
                        weight_decay,
                        scope,
                        use_bias=True):
    with tf.variable_scope(scope):
        kernel = get_kernel(input_tensor, 1, out_depth, weight_decay, "custom", init_factor)
        out_tensor = conv(input_tensor, kernel, 1, use_bias, "VALID")
        if not tf.executing_eagerly():
            tf.add_to_collection("monitor", {out_tensor.name.encode("ascii"): tf.math.reduce_variance(out_tensor)})
    return tf.squeeze(out_tensor, axis=[1, 2])
