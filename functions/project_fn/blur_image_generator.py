import tensorflow as tf
from functions.project_fn.utils import get_tensor_shape
from functions.project_fn import utils
from functions.project_fn.preprocess_developing import add_gaussian_noise, rgb_perturb, random_quality


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


def create_blur_images_individual(parsed_data, config):
    # intended to augment individual images within clone batch
    new_parsed_data = []

    decode_features = {"kernel": tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                       "h": tf.FixedLenFeature([], tf.int64),
                       "w": tf.FixedLenFeature([], tf.int64)}

    def blur_kernel_getter(tfrecord, number):
        def parser(data):
            parsed = tf.parse_single_example(data, decode_features)
            kernel_data = tf.convert_to_tensor(parsed["kernel"])
            h = tf.cast(tf.convert_to_tensor(parsed["h"]), tf.int32)
            w = tf.cast(tf.convert_to_tensor(parsed["w"]), tf.int32)
            kernel_data = tf.reshape(kernel_data, [h, w])

            rnd_angle = tf.random.uniform(shape=[], minval=0.0, maxval=360.0)

            role_dice = tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32)
            rnd_scale = tf.random.uniform(shape=[], minval=1.0, maxval=1.4)
            new_h = tf.cast(tf.cast(h, tf.float32) * rnd_scale, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) * rnd_scale, tf.int32)
            # dice 0 for doing nothing
            # dice 1 for random rotate
            # dice 2 for random scaling
            kernel = tf.cond(tf.equal(role_dice, 1), lambda: tf.contrib.image.rotate(kernel_data, rnd_angle, interpolation="BILINEAR"), lambda: kernel_data)
            kernel = tf.cond(tf.equal(role_dice, 2), lambda: tf.image.resize_bicubic(kernel[tf.newaxis, :, :, tf.newaxis], [new_h, new_w]), lambda: kernel)
            kernel = tf.squeeze(kernel)
            kernel = utils.remap_kernel_to_fixed_odd_squre(kernel_data, 33)
            # kernel = tf.contrib.image.rotate(kernel, rnd_angle, interpolation="BILINEAR")
            return kernel / tf.reduce_sum(tf.squeeze(kernel))

        if tfrecord:
            return tf.data.TFRecordDataset(tfrecord).repeat().shuffle(config.batch_size * 40).map(parser).batch(number).make_one_shot_iterator().get_next()
        else:
            return None

    b1 = int(config.batch_size * 0.7)  # int(config.batch_size * 0.6) #random trajectory
    b2 = 0  # int(config.batch_size * 0.1)
    b3 = 0  # int(config.batch_size * 0.1)
    b4 = config.batch_size - b1  # int(config.batch_size * 0.1) #gaussian
    b5 = 0  # config.batch_size - b1 - b2 - b3 - b4

    blur_kernel_bank = [blur_kernel_getter(config.blind_kernel_dir, b1),
                        blur_kernel_getter(config.box_kernel_dir, b2),
                        blur_kernel_getter(config.circle_kernel_dir, b3),
                        blur_kernel_getter(config.gaussian_kernel_dir, b4),
                        blur_kernel_getter(config.line_kernel_dir, b5)]
    blur_kernel_bank = [blur_kernel for blur_kernel in blur_kernel_bank if blur_kernel is not None]
    blur_kernel_bank = tf.concat(blur_kernel_bank, 0)
    blur_kernel_bank = tf.random.shuffle(blur_kernel_bank)
    blur_kernel_bank = tf.split(blur_kernel_bank, len(config.gpu_ids), 0)

    for i, gpu_id in enumerate(config.gpu_ids):
        sharp_images = parsed_data["image"]
        clone_batch, h, w, c = get_tensor_shape(sharp_images)
        corrupted_imgs = []
        with tf.device("/device:GPU:" + str(gpu_id)):
            _, h, w, _ = get_tensor_shape(sharp_images)
            for idx in range(clone_batch):
                blur_kernel = tf.concat([blur_kernel_bank[gpu_id][idx, ::][:, :, tf.newaxis, tf.newaxis]] * c, 2)
                sharp_img = sharp_images[idx, ::][tf.newaxis, ::]

                # apply blur_kernel
                do_blur = tf.less_equal(tf.random_uniform([]), config.blur_aug_probability)
                corrupted_img = tf.cond(do_blur, lambda: tf.nn.depthwise_conv2d(sharp_img, blur_kernel, [1, 1, 1, 1], "SAME"), lambda: sharp_img)

                if config.gaussian_noise_prob > 0.0:
                    # todo: do gaussian noise again or not?
                    do_gaussian_noise = tf.less_equal(tf.random.uniform([], maxval=1.0, dtype=tf.float32), config.gaussian_noise_prob)
                    stddev_min = config.gaussian_noise_std[0]
                    stddev_max = config.gaussian_noise_std[1]
                    corrupted_img = tf.cond(do_gaussian_noise, lambda: add_gaussian_noise(corrupted_img, stddev_min, stddev_max), lambda: corrupted_img)
                corrupted_img.set_shape([1, h, w, c])
                corrupted_imgs.append(corrupted_img)
            corrupted_imgs = tf.concat(corrupted_imgs, 0)
            corrupted_imgs = center_crop(corrupted_imgs, config)
        corrupted_imgs.set_shape([clone_batch, None, None, c])
        new_parsed_data.append({"image": corrupted_imgs, "gt": parsed_data["gt"], "tf_kernel": blur_kernel_bank[gpu_id]})
    return new_parsed_data
