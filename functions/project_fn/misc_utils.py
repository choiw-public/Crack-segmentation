from collections import OrderedDict
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import math
import imp
import cv2 as cv
import glob
import os
import re


def get_tensor_shape(tensor):
    _static_shape = tensor.get_shape().as_list()
    _dynamic_shape = tf.unstack(tf.shape(tensor))
    _dims = [s[1] if s[0] is None else s[0] for s in zip(_static_shape, _dynamic_shape)]
    return _dims


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
        file_list = natsorted(file_list)
    return file_list


def get_all_ckpt_id(config):
    ckpt_full_list = glob.glob(os.path.join(config.ckpt_dir, "model_step*"))
    all_ckpt_id = []
    for filename in ckpt_full_list:
        basename = os.path.basename(filename)
        if len(basename.split(".")) != 1:
            all_ckpt_id.append(basename.split(".")[0])
    all_ckpt_id = list(set(all_ckpt_id))
    all_ckpt_id.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])
    return all_ckpt_id


def get_all_ckpt_list(config):
    all_ckpt_id = get_all_ckpt_id(config)
    return [os.path.join(config.ckpt_dir, ckpt_id) for ckpt_id in all_ckpt_id]


def get_ckpt_list_in_range(config):
    all_ckpt_id = get_all_ckpt_id(config)
    if config.ckpt_start == "beginning":
        start_idx = 0
    else:
        start_idx = all_ckpt_id.index("model_step-%d" % config.ckpt_start)

    if config.ckpt_end == "end":
        end_idx = None
    else:
        end_idx = all_ckpt_id.index("model_step-%d" % config.ckpt_end) + 1
    all_ckpt_id = all_ckpt_id[start_idx:end_idx:config.ckpt_step]
    return [os.path.join(config.ckpt_dir, ckpt_id) for ckpt_id in all_ckpt_id]


def get_ckpt(config):
    all_ckpt_id = get_all_ckpt_id(config)
    ckpt_idx = all_ckpt_id.index("model_step-%d" % config.ckpt_id)
    ckpt_id = all_ckpt_id[ckpt_idx]
    return os.path.join(config.ckpt_dir, ckpt_id)


def pad_for_stride(tensor, config):
    if config.task == "segmentation":
        stride = config.num_stride

        def update_length(_length, _stride):
            _length = tf.cast(_length, tf.float32)
            should_apply_update = tf.not_equal(tf.floormod(0.5 ** _stride * (_length - 1), 1), 0.0)
            return tf.cond(should_apply_update, lambda: tf.cast(tf.ceil(_length / (2 ** _stride)) * 2 ** _stride + 1, tf.int32), lambda: tf.cast(_length, tf.int32))

        n, h, w, c = get_tensor_shape(tensor)

        new_h = update_length(h, stride)
        new_w = update_length(w, stride)

        h_center = h / 2
        w_center = w / 2
        new_h_center = new_h / 2
        new_w_center = new_w / 2

        num_pad_h1 = new_h_center - h_center
        num_pad_h2 = new_h - h - num_pad_h1
        num_pad_w1 = new_w_center - w_center
        num_pad_w2 = new_w - w - num_pad_w1
        num_pad_cache = [[0, 0], [num_pad_h1, num_pad_h2], [num_pad_w1, num_pad_w2], [0, 0]]
        aligned_tensor = tf.pad(tensor, num_pad_cache)
        return aligned_tensor, num_pad_cache[1:3]
    elif config.task == "deblur":
        return tensor, None
    else:
        raise ValueError("not supported")


def unpad_for_stride(tensor, pad_cache):
    if pad_cache:
        if not len(get_tensor_shape(tensor)) == 4:
            raise ValueError("the rank of tensor must be 2, 3, or 4")
        _, h, w, c = get_tensor_shape(tensor)
        h1 = pad_cache[0][0]
        h2 = h - pad_cache[0][1]
        w1 = pad_cache[1][0]
        w2 = w - pad_cache[1][1]
        return tensor[:, h1:h2, w1:w2, :]
    else:
        return tensor


def count_trainable():
    all_trainables = tf.trainable_variables()
    parameters = 0
    for variable in all_trainables:
        parameters += np.prod([int(para) for para in variable.get_shape])
    print(parameters)


def gaussian_kernel_2d(size, sigma):
    # using tensorflow
    distribution = tfp.distributions.Normal(0.0, sigma)
    size = tf.cast(size, tf.float32)
    x = tf.linspace(-(size // 2), size // 2, tf.cast(size, tf.int32))
    vals = distribution.prob(x)
    kernel = tf.einsum("i,j->ij", vals, vals)
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel[:, :, tf.newaxis, tf.newaxis]


def numpy_gaussian_kernel_2d(size, sigma):
    # using numpy
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def get_random_gaussian_kernel_size(target_tensor):
    _, h, w, c = get_tensor_shape(target_tensor)
    # if not c == 1:
    #     raise ValueError("currently grayscale deblur is only supported")

    length = tf.reduce_min([h, w])
    if length % 2 == 0:
        raise ValueError("this function is designed to apply an odd size of feature map")

    max_size = tf.reduce_min([tf.cast(length, tf.float32), 51.0])
    max_size = tf.cast(tf.cond(tf.equal(max_size % 2.0, 0.0), lambda: max_size - 1, lambda: max_size), tf.int32)
    # size_table = tf.range(min_size, max_size + 2, 2)
    size_table = tf.range(5, max_size + 2, 2)
    rnd_idx = tf.random_uniform([], maxval=get_tensor_shape(size_table)[0], dtype=tf.int32)
    return size_table[rnd_idx]


def get_random_sigmas(size, num_scales):
    sigmas = []
    size = tf.cast(size, tf.float32)
    for i in range(num_scales):
        min_fwhm = tf.divide(size, 5.)
        max_fwhm = tf.divide(tf.multiply(size, 2.0), 3.0)
        fwhm = tf.random_uniform([], minval=min_fwhm, maxval=max_fwhm)
        sigmas.append(fwhm / 2.35482)
    sigmas = tf.contrib.framework.sort(tf.stack(sigmas))
    return sigmas


def gen_gaussian_kernel_by_sigma(size, sigma):
    # return numpy_gaussian_kernel_2d(size, sigma)
    return gaussian_kernel_2d(size, sigma)


def gen_gaussian_kernel_by_random_sigma(size):
    min_fwhm = size / 5.
    max_fwhm = size * 2. / 3
    fwhm = np.random.uniform(min_fwhm, max_fwhm)
    sigma = fwhm / np.sqrt(8 * np.log(2))
    return numpy_gaussian_kernel_2d(size, sigma)


def gen_box_kernel(size):
    kernel = tf.ones([size, size, 1, 1])
    return kernel / (tf.reduce_sum(kernel))


def slice_by_kernel_size(tensor, kernel_size):
    _, h, w, _ = get_tensor_shape(tensor)
    return tensor[:, kernel_size:h - kernel_size, kernel_size:w - kernel_size, :]


def coefficient_of_determination(img1, img2):
    _, h, w, _ = get_tensor_shape(img1)
    mu_img1 = tf.reduce_mean(img1)
    mu_img2 = tf.reduce_mean(img2)
    img1_term = img1 - mu_img1
    img2_term = img2 - mu_img2
    covariance = tf.reduce_sum(img1_term * img2_term)
    prod_std1 = tf.sqrt(tf.reduce_sum(img1_term ** 2))
    prod_std2 = tf.sqrt(tf.reduce_sum(img2_term ** 2))
    pearson_coefficient = covariance / (prod_std1 * prod_std2)
    return pearson_coefficient ** 2


def get_random_kernel_sizes(img1, num_kernels):
    random_kernel_sizes = []
    for i in range(num_kernels):
        random_kernel_sizes.append(get_random_gaussian_kernel_size(img1))
    return tf.contrib.framework.sort(tf.stack(random_kernel_sizes))


def remap_kernel(kernel, kernel_size, reference_kernel_size):
    # this function is intended to zero pad "kernel" to have size of "reference_kernel"
    # "kernel" must have smaller dimension than "reference_kernel"
    num_zeros = (reference_kernel_size - kernel_size) / 2
    pad = [[num_zeros, num_zeros], [num_zeros, num_zeros], [0, 0], [0, 0]]
    return tf.pad(kernel, pad)


def remap_kernel_to_fixed_odd_squre(kernel, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size should be an odd number")

    kernel_shape = get_tensor_shape(kernel)
    is_odd = tf.cast(tf.floormod(kernel_size, 2), tf.bool)

    num_pad_h1 = (kernel_size - kernel_shape[0]) / 2
    num_pad_h2 = kernel_size - num_pad_h1 - kernel_shape[0]

    num_pad_w1 = (kernel_size - kernel_shape[1]) / 2
    num_pad_w2 = kernel_size - num_pad_w1 - kernel_shape[1]

    pad = [[num_pad_h1, num_pad_h2], [num_pad_w1, num_pad_w2]]
    return tf.pad(kernel, pad)


def remap_kernel_to_odd_squre(kernel):
    kernel_shape = get_tensor_shape(kernel)

    kernel_size = tf.reduce_max(kernel_shape)
    is_odd = tf.cast(tf.floormod(kernel_size, 2), tf.bool)
    kernel_size = tf.cond(is_odd, lambda: kernel_size, lambda: kernel_size + 1)

    num_pad_h1 = (kernel_size - kernel_shape[0]) / 2
    num_pad_h2 = kernel_size - num_pad_h1 - kernel_shape[0]

    num_pad_w1 = (kernel_size - kernel_shape[1]) / 2
    num_pad_w2 = kernel_size - num_pad_w1 - kernel_shape[1]

    pad = [[num_pad_h1, num_pad_h2], [num_pad_w1, num_pad_w2]]
    return tf.pad(kernel, pad)


def sobel(gaussian_filtered):
    sobel = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x = tf.reshape(sobel, [3, 3, 1, 1])
    sobel_y = tf.transpose(sobel_x, [1, 0, 2, 3])
    sobel_x = tf.concat([sobel_x, sobel_x, sobel_x], 2)
    sobel_y = tf.concat([sobel_y, sobel_y, sobel_y], 2)

    sobel_filtered_x = tf.nn.depthwise_conv2d(gaussian_filtered, sobel_x, [1, 1, 1, 1], "SAME")
    sobel_filtered_y = tf.nn.depthwise_conv2d(gaussian_filtered, sobel_y, [1, 1, 1, 1], "SAME")
    return tf.sqrt(sobel_filtered_x ** 2 + sobel_filtered_y ** 2)


def add_to_collection(featuremap, has_kernel=True):
    tnsr_name = featuremap.name.encode("ascii")
    filter_name = os.path.join(*tnsr_name.split("/")[1:-1])
    trainable = []
    if has_kernel:
        for kernel in tf.trainable_variables():
            if filter_name in kernel.name:
                trainable.append(kernel)
        tf.add_to_collection("layer", [trainable, featuremap])
    else:
        tf.add_to_collection("layer", [featuremap])


def init_check(sess):
    layer_collection = tf.get_collection("layer")

    # To see histogram of input data, uncomment the below and manually check
    # in_container = []
    # tf_in = layer_collection[0][0]
    # for i in range(100):
    #     in_container.append(sess.run(tf_in))
    # in_data = np.concatenate(in_container, 0)
    # plt.hist(in_data.flatten(), bins="auto")
    layers = sess.run(layer_collection)
    with open("layer_statistics.csv", "w") as w:
        w.write("layer,filter_mean,filter_var,filter_max, filter_min, fmap_mean, fmap_var, fmap_max, fmap_min \n")
        for i, layer in enumerate(layers):
            layer_name = layer_collection[i][-1].name.encode("ascii")
            if len(layer) == 1:  # in case there are no trainable filters
                fmap = layer[0]
                w.write("%s, -, -, -, -, %.3f, %.3f, %.3f, %.3f\n" % (layer_name, fmap.mean(), fmap.var(), fmap.max(), fmap.min()))
            else:
                filter = layer[0][0]
                bias = layer[0][1]
                fmap = layer[1]
                w.write("%s, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n" % (
                    layer_name, filter.mean(), filter.var(), filter.max(), filter.min(), fmap.mean(), fmap.var(), fmap.max(), fmap.min()))


def categorized_ssim(config):
    dir_name = config.eval_log_dir
    eval_list = list_getter(dir_name, "csv", "model_step-")
    head_cache = []
    with open(os.path.join(dir_name, "00.metric_overall_categorized.csv"), "w") as w:
        for m in eval_list:
            with open(m, "r") as r:
                eval_metric = r.readlines()
            step = int(m[m.find("(") + 1: m.find(")")].split("-")[-1])
            categorized = dict()
            for e in eval_metric[1:]:
                category = "SSIM" + "_" + e[e.find("(") + 1:e.find(")")]
                score = float(e.split(",")[-1].replace(" ", "").replace("\n", ""))
                if category in categorized.keys():
                    categorized[category][0] += score
                    categorized[category][1] += 1
                else:
                    categorized[category] = [score, 1]
            for k, v in categorized.iteritems():
                categorized[k] = v[0] / float(v[1])

            if head_cache:
                new_head = categorized.keys()
                new_head.sort()
                if head_cache == new_head:
                    pass
                else:
                    raise ValueError("Different SSIM range. check '%s' file" % m)
            else:
                head_cache = categorized.keys()
                head_cache.sort()
                w.write("ckpt_step, ")
                w.write(", ".join([head for head in head_cache]) + "\n")

            w.write("%s, " % step)
            w.write(", ".join([str(categorized[h]) for h in head_cache]) + "\n")


def detect_ridge(input_tensor):
    dx_kernel = tf.constant([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], tf.float32)
    dx_kernel = dx_kernel[:, :, tf.newaxis, tf.newaxis]
    dy_kernel = tf.constant([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], tf.float32)
    dy_kernel = dy_kernel[:, :, tf.newaxis, tf.newaxis]
    dy = tf.nn.conv2d(input_tensor, tf.concat([dy_kernel] * 3, 2), [1, 1, 1, 1], padding="SAME")
    dx = tf.nn.conv2d(input_tensor, tf.concat([dx_kernel] * 3, 2), [1, 1, 1, 1], padding="SAME")
    dydy = tf.nn.conv2d(dy, dy_kernel, [1, 1, 1, 1], padding="SAME")
    dydx = tf.nn.conv2d(dy, dx_kernel, [1, 1, 1, 1], padding="SAME")
    dxdx = tf.nn.conv2d(dx, dx_kernel, [1, 1, 1, 1], padding="SAME")
    eq1 = (dydy + dxdx) / 2
    eq2 = tf.sqrt(4 * dydx ** 2 + (dydy - dxdx) ** 2) / 2
    ridge_maxima = eq1 + eq2
    # ridge_minima = eq1 - eq2
    return tf.maximum(ridge_maxima, 0)


def detect_ridge_channel_wise(input_tensor):
    _, _, _, c = get_tensor_shape(input_tensor)
    dx_kernel = tf.constant([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], tf.float32)
    dx_kernel = dx_kernel[:, :, tf.newaxis, tf.newaxis]
    dy_kernel = tf.constant([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], tf.float32)
    dy_kernel = dy_kernel[:, :, tf.newaxis, tf.newaxis]
    dx_kernel = tf.concat([dx_kernel] * c, 2)
    dy_kernel = tf.concat([dy_kernel] * c, 2)
    dy = tf.nn.depthwise_conv2d(input_tensor, dy_kernel, [1, 1, 1, 1], padding="SAME")
    dx = tf.nn.depthwise_conv2d(input_tensor, dx_kernel, [1, 1, 1, 1], padding="SAME")
    dydy = tf.nn.depthwise_conv2d(dy, dy_kernel, [1, 1, 1, 1], padding="SAME")
    dydx = tf.nn.depthwise_conv2d(dy, dx_kernel, [1, 1, 1, 1], padding="SAME")
    dxdx = tf.nn.depthwise_conv2d(dx, dx_kernel, [1, 1, 1, 1], padding="SAME")
    eq1 = (dydy + dxdx) / 2
    eq2 = tf.sqrt(4 * dydx ** 2 + (dydy - dxdx) ** 2) / 2
    ridge_maxima = eq1 + eq2
    # ridge_minima = eq1 - eq2
    return tf.maximum(ridge_maxima, 0)


def tensor_summary(layer_collection, layers):
    with open("layer_statistics.csv", "w") as w:
        w.write("layer,filter_mean,filter_var,filter_max, filter_min, fmap_mean, fmap_var, fmap_max, fmap_min \n")
        for i, layer in enumerate(layers):
            layer_name = layer_collection[i][-1].name.encode("ascii")
            if len(layer) == 1:  # in case there are no trainable filters
                fmap = layer[0]
                w.write("%s, -, -, -, -, %.3f, %.3f, %.3f, %.3f\n" % (layer_name, fmap.mean(), fmap.var(), fmap.max(), fmap.min()))
            else:
                f = layer[0][0]  # filter
                b = layer[0][1]  # bias
                fmap = layer[1]  # featuremap
                w.write("%s, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n" % (
                    layer_name, f.mean(), f.var(), f.max(), f.min(), fmap.mean(), fmap.var(), fmap.max(), fmap.min()))
    w.close()


def skeletonize(binary_image):
    binary_image = binary_image.copy()
    skel = binary_image.copy()
    skel[:, :] = 0
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv.morphologyEx(binary_image, cv.MORPH_ERODE, kernel)
        temp = cv.morphologyEx(eroded, cv.MORPH_DILATE, kernel)
        temp = cv.subtract(binary_image, temp)
        skel = cv.bitwise_or(skel, temp)
        binary_image[:, :] = eroded[:, :]
        if cv.countNonZero(binary_image) == 0:
            break

    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(skel, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 10
    skel2 = np.zeros(output.get_shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            skel2[output == i + 1] = 1
    return skel2


def get_crop_coordinates(h_cnt, w_cnt, src_img, crop_size):
    src_img_h, src_img_w, _ = src_img.get_shape
    _h1 = h_cnt - (crop_size - 1) / 2
    if _h1 < 0:
        _h1 = 0
    elif _h1 > src_img_h - crop_size:
        _h1 = src_img_h - crop_size
    _h2 = _h1 + crop_size

    _w1 = w_cnt - (crop_size - 1) / 2
    if _w1 < 0:
        _w1 = 0
    elif _w1 > src_img_w - crop_size:
        _w1 = src_img_w - crop_size
    _w2 = _w1 + crop_size
    return _h1, _h2, _w1, _w2

# tf.enable_eager_execution()
# tensor = tf.random_uniform([1, 257, 247, 1], maxval=255)
# np_kernel = numpy_gaussian_kernel_2d(11, 1.5)
# tf_kernel = tf_gaussian_kernel_2d(11, 1.5)
