import tensorflow as tf
from functions.project_fn.misc_utils import get_tensor_shape
from functions.project_fn import misc_utils


def get_random_kernel(blur_kernel_bank, config):
    blur_kernel_bank = dict((k, v) for k, v in blur_kernel_bank.iteritems() if v is not None)

    def get_kernel_index():
        return tf.random_uniform((), minval=0, maxval=len(blur_kernel_bank.items()), dtype=tf.int32)

    if config.task == "deblur":
        kernel_type_index = get_kernel_index()
    elif config.task == "segmentation":
        kernel_type_index = tf.cond(tf.less_equal(tf.random_uniform([], maxval=1.0), config.blur_aug_probability), lambda: get_kernel_index(), lambda: 99)
    else:
        raise ValueError("not supported task")

    dummy_kernel = tf.ones([1, 1], dtype=tf.float32)  # this kernel returns the same output of input
    dummy_kernel_name = tf.constant("as_is")
    for i, (key, value) in enumerate(blur_kernel_bank.iteritems()):
        if value is not None:
            if i == 0:
                kernel_name = tf.cond(tf.equal(kernel_type_index, i), lambda: tf.constant(key), lambda: dummy_kernel_name)
            else:
                kernel_name = tf.cond(tf.equal(kernel_type_index, i), lambda: tf.constant(key), lambda: kernel_name)

    for i, (key, value) in enumerate(blur_kernel_bank.iteritems()):
        if i == 0:
            random_kernel = tf.cond(tf.equal(kernel_name, key), lambda: tf.squeeze(value), lambda: dummy_kernel)
        else:
            random_kernel = tf.cond(tf.equal(kernel_name, key), lambda: tf.squeeze(value), lambda: random_kernel)
    random_kernel.set_shape([None, None])
    return {"tf_kernel": random_kernel, "kernel_name": kernel_name}


def center_crop(input_tensor, config):
    crop_size = config.crop_size
    input_size = get_tensor_shape(input_tensor)
    center_height = (input_size[1] - 1) / 2
    center_width = (input_size[2] - 1) / 2
    h1 = center_height - (crop_size[0] - 1) / 2
    h2 = h1 + crop_size[0]
    w1 = center_width - (crop_size[1] - 1) / 2
    w2 = w1 + crop_size[1]
    return input_tensor[:, h1:h2, w1:w2, :]


def create_blur_images_individual(data_for_clones, config):
    # intended to augment individual images within clone batch
    new_data_for_clones = []

    decode_features = {"kernel": tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                       "h": tf.FixedLenFeature([], tf.int64),
                       "w": tf.FixedLenFeature([], tf.int64)}

    def blur_kernel_getter(tfrecord):
        def parser(data):
            parsed = tf.parse_single_example(data, decode_features)
            kernel_data = tf.convert_to_tensor(parsed["kernel"])
            h = tf.convert_to_tensor(parsed["h"])
            w = tf.convert_to_tensor(parsed["w"])
            return tf.reshape(kernel_data, [h, w])

        if tfrecord:
            return tf.data.TFRecordDataset(tfrecord).repeat().shuffle(config.batch_size * 40).map(parser).make_one_shot_iterator().get_next()
        else:
            return None

    blur_kernel_bank = {"blind_kernel": blur_kernel_getter(config.blind_kernel_dir),
                        "box_kernel": blur_kernel_getter(config.box_kernel_dir),
                        "circle_kernel": blur_kernel_getter(config.circle_kernel_dir),
                        "gaussian_kernel": blur_kernel_getter(config.gaussian_kernel_dir),
                        "line_kernel": blur_kernel_getter(config.line_kernel_dir)}
    for i, gpu_id in enumerate(config.gpu_ids):
        if config.task == "deblur":
            sharp_images = data_for_clones[i]["gt"]
        elif config.task == "segmentation":
            sharp_images = data_for_clones[i]["input"]
        else:
            raise ValueError("not supported task")
        clone_batch, _, _, c = get_tensor_shape(sharp_images)
        blur_images = []
        blur_kernels = []
        for idx in range(clone_batch):
            with tf.device("/device:CPU:0"):
                blur_kernel = get_random_kernel(blur_kernel_bank, config)
                blur_kernel["tf_kernel"] = misc_utils.remap_kernel_to_odd_squre(blur_kernel["tf_kernel"])
                blur_kernel["tf_kernel"] = tf.concat([tf.expand_dims(tf.expand_dims(blur_kernel["tf_kernel"], 2), 3)] * c, 2)
                blur_kernels.append(blur_kernel)
            with tf.device("/device:GPU:" + str(gpu_id)):
                blur_images.append(tf.nn.depthwise_conv2d(sharp_images[idx, ::][tf.newaxis, ::], blur_kernel["tf_kernel"], [1, 1, 1, 1], "SAME"))
        blur_images = tf.concat(blur_images, 0)

        blur_images = center_crop(blur_images, config)
        if config.task == "deblur":
            ground_truth = center_crop(sharp_images, config)
        elif config.task == "segmentation":
            ground_truth = center_crop(data_for_clones[i]["gt"], config)
        else:
            raise ValueError("not supported task")

        blur_images.set_shape([clone_batch, None, None, c])
        if config.task == "deblur":
            new_data_for_clones.append({"input": blur_images, "gt": ground_truth, "filename": data_for_clones[i]["filename"], "tf_kernel": blur_kernels})
        elif config.task == "segmentation":
            new_data_for_clones.append({"input": blur_images, "gt": ground_truth, "filename": data_for_clones[i]["filename"], "tf_kernel": blur_kernels})
    return new_data_for_clones
