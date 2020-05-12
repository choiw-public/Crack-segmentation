# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions related to preprocessing inputs."""
import tensorflow as tf
import glob
import cv2 as cv

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_tensor_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)]
    return dims


def flip_or_rotate(tensor_list):
    flip_prob = tf.constant(FLAGS.prob_flip)
    rotate_prob = tf.constant(FLAGS.prob_rotate)

    def flip():
        flip_dim = 1
        flipped = []
        for tensor in tensor_list:
            if flip_dim < 0 or flip_dim >= len(tensor.get_shape().as_list()):
                raise ValueError('dim must represent a valid dimension.')
            flipped.append(tf.reverse_v2(tensor, [flip_dim]))
        return flipped

    def rotate():
        rotate_k = tf.random_uniform((), maxval=4, dtype=tf.int32)
        rotated = []
        for tensor in tensor_list:
            if len(tensor.get_shape()) == 2:
                tensor = tf.expand_dims(tensor, 2)
                rotated.append(tf.squeeze(tf.image.rot90(tensor, rotate_k)))
            else:
                rotated.append(tf.image.rot90(tensor, rotate_k))
        return rotated

    if FLAGS.img_flip and FLAGS.job == 'train':
        is_flipped = tf.less_equal(tf.random_uniform([]), flip_prob)
        return tf.cond(is_flipped, flip, lambda: tensor_list)
    elif FLAGS.img_rotate and FLAGS.job == 'train':
        is_rotate = tf.less_equal(tf.random_uniform([]), rotate_prob)
        return tf.cond(is_rotate, rotate, lambda: tensor_list)
    else:
        return tensor_list[0]  # TODO: this is not valid if tensor_list has more than 1 elements.


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """Pads the given image with the given pad_value.

    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes are not
    known during graph construction.

    Args:
      image: 3-D tensor with original_shape [height, width, channels]
      offset_height: Number of rows of zeros to add on top.
      offset_width: Number of columns of zeros to add on the left.
      target_height: Height of output image.
      target_width: Width of output image.
      pad_value: Value to pad the image tensor with.

    Returns:
      3-D tensor of original_shape [target_height, target_width, channels].

    Raises:
      ValueError: If the original_shape of image is incompatible with the offset_* or
      target_* arguments.
    """
    image_rank = tf.rank(image)
    image_rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong image tensor rank [Expected] [Actual]',
         3, image_rank])
    with tf.control_dependencies([image_rank_assert]):
        image -= pad_value
    image_shape = tf.shape(image)
    height, width = image_shape[0], image_shape[1]
    target_width_assert = tf.Assert(
        tf.greater_equal(
            target_width, width),
        ['target_width must be >= width'])
    target_height_assert = tf.Assert(
        tf.greater_equal(target_height, height),
        ['target_height must be >= height'])
    with tf.control_dependencies([target_width_assert]):
        after_padding_width = target_width - offset_width - width
    with tf.control_dependencies([target_height_assert]):
        after_padding_height = target_height - offset_height - height
    offset_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(after_padding_width, 0),
            tf.greater_equal(after_padding_height, 0)),
        ['target size not possible with the given target offsets'])

    height_params = tf.stack([offset_height, after_padding_height])
    width_params = tf.stack([offset_width, after_padding_width])
    channel_params = tf.stack([0, 0])
    with tf.control_dependencies([offset_assert]):
        paddings = tf.stack([height_params, width_params, channel_params])
    padded = tf.pad(image, paddings)
    return padded + pad_value


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
      image: an image of original_shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      The cropped (and resized) image.

    Raises:
      ValueError: if `image` doesn't have rank of 3.
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
        raise ValueError('input must have rank of 3')
    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image


def random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:

      image, depths, normals = random_crop([image, depths, normals], 120, 150)

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.

    Returns:
      the sharp_image_list with cropped images.

    Raises:
      ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty sharp_image_list.')
    image_list = [img for img in image_list if img != None]

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    out_list = [_crop(image, offset_height, offset_width,
                      crop_height, crop_width) for image in image_list]
    if len(out_list) == 2:
        return out_list[0], out_list[1]
    elif len(out_list) == 1:
        return out_list[0], None
    else:
        raise ValueError('This function is in developing. Check input of this function')


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.

    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.

    Returns:
      A random scale value selected between minimum and maximum value.

    Raises:
      ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.to_float(min_scale_factor)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label=None, scale=1.0):
    """Randomly scales image and label.

    Args:
      image: Image with original_shape [height, width, 3].
      label: Label with original_shape [height, width, 1].
      scale: The value to scale image and label.

    Returns:
      Scaled image and label.
    """
    # No random scaling if scale == 1.
    if scale == 1.0:
        return image, label
    image_shape = tf.shape(image)
    new_dim = tf.to_int32(tf.to_float([image_shape[0], image_shape[1]]) * scale)

    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = tf.squeeze(tf.image.resize_bilinear(
        tf.expand_dims(image, 0),
        new_dim,
        align_corners=True), [0])
    if label is not None:
        label = tf.squeeze(tf.image.resize_nearest_neighbor(
            tf.expand_dims(label, 0),
            new_dim,
            align_corners=True), [0])

    return image, label


def resolve_shape(tensor, rank=None, scope=None):
    """Fully resolves the original_shape of a Tensor.

    Use as much as possible the original_shape components already known during graph
    creation and resolve the remaining ones during runtime.

    Args:
      tensor: Input tensor whose original_shape we query.
      rank: The rank of the tensor, provided that we know it.
      scope: Optional img_name scope.

    Returns:
      original_shape: The full original_shape of the tensor.
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()

        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

        return shape


def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None,
                    align_corners=True,
                    label_layout_is_chw=False,
                    scope=None,
                    method=tf.image.ResizeMethod.BILINEAR):
    """Resizes image or label so their sides are within the provided range.

    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum size is equal to min_size
       without the other side exceeding max_size, then do so.
    2. Otherwise, resize so the largest side is equal to max_size.

    An integer in `range(factor)` is added to the computed sides so that the
    final dimensions are multiples of `factor` plus one.

    Args:
      image: A 3D tensor of original_shape [height, width, channels].
      label: (optional) A 3D tensor of original_shape [height, width, channels] (default)
        or [channels, height, width] when label_layout_is_chw = True.
      min_size: (scalar) desired size of the smaller image side.
      max_size: (scalar) maximum allowed size of the larger image side. Note
        that the output dimension is no larger than max_size and may be slightly
        smaller than min_size when factor is not None.
      factor: Make output size multiple of factor plus one.
      align_corners: If True, exactly align all 4 corners o input and output.
      label_layout_is_chw: If true, the label has original_shape [channel, height, width].
        We support this case because for some instance segmentation dataset, the
        instance segmentation is saved as [num_instances, height, width].
      scope: Optional img_name scope.
      method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.

    Returns:
      A 3-D tensor of original_shape [new_height, new_width, channels], where the image
      has been resized (with the specified method) so that
      min(new_height, new_width) == ceil(min_size) or
      max(new_height, new_width) == ceil(max_size).

    Raises:
      ValueError: If the image is not a 3D tensor.
    """
    with tf.name_scope(scope, 'resize_to_range', [image]):
        new_tensor_list = []
        min_size = tf.to_float(min_size)
        if max_size is not None:
            max_size = tf.to_float(max_size)
            # Modify the max_size to be a multiple of factor plus 1 and make sure the
            # max dimension after resizing is no larger than max_size.
            if factor is not None:
                max_size = (max_size + (factor - (max_size - 1) % factor) % factor
                            - factor)

        [orig_height, orig_width, _] = resolve_shape(image, rank=3)
        orig_height = tf.to_float(orig_height)
        orig_width = tf.to_float(orig_width)
        orig_min_size = tf.minimum(orig_height, orig_width)

        # Calculate the larger of the possible sizes
        large_scale_factor = min_size / orig_min_size
        large_height = tf.to_int32(tf.ceil(orig_height * large_scale_factor))
        large_width = tf.to_int32(tf.ceil(orig_width * large_scale_factor))
        large_size = tf.stack([large_height, large_width])

        new_size = large_size
        if max_size is not None:
            # Calculate the smaller of the possible sizes, use that if the larger
            # is too big.
            orig_max_size = tf.maximum(orig_height, orig_width)
            small_scale_factor = max_size / orig_max_size
            small_height = tf.to_int32(tf.ceil(orig_height * small_scale_factor))
            small_width = tf.to_int32(tf.ceil(orig_width * small_scale_factor))
            small_size = tf.stack([small_height, small_width])
            new_size = tf.cond(
                tf.to_float(tf.reduce_max(large_size)) > max_size,
                lambda: small_size,
                lambda: large_size)
        # Ensure that both output sides are multiples of factor plus one.
        if factor is not None:
            new_size += (factor - (new_size - 1) % factor) % factor
        new_tensor_list.append(tf.image.resize_images(
            image, new_size, method=method, align_corners=align_corners))
        if label is not None:
            if label_layout_is_chw:
                # Input label has original_shape [channel, height, width].
                resized_label = tf.expand_dims(label, 3)
                resized_label = tf.image.resize_nearest_neighbor(
                    resized_label, new_size, align_corners=align_corners)
                resized_label = tf.squeeze(resized_label, 3)
            else:
                # Input label has original_shape [height, width, channel].
                resized_label = tf.image.resize_images(
                    label, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                    align_corners=align_corners)
            new_tensor_list.append(resized_label)
        else:
            new_tensor_list.append(None)
        return new_tensor_list


def random_brightness(_image_):
    return tf.image.random_brightness(_image_, FLAGS.brightness_constant)


def random_contrast(_image_):
    # return tf.image.adjust_contrast(_image_, 4.0)
    return tf.image.random_contrast(_image_, FLAGS.contrast_constant[0], FLAGS.contrast_constant[1])


def random_hue(_image_):
    return tf.image.random_hue(_image_, FLAGS.hue_constant)


def random_saturation(_image_):
    return tf.image.random_saturation(_image_, FLAGS.saturation_constant[0],
                                      FLAGS.saturation_constant[1])


def add_gaussian_noise(_image_):
    _image_ = tf.to_float(_image_) / 255.0
    noise = tf.random_normal(shape=tf.shape(_image_), mean=0.0, stddev=FLAGS.gaussian_noise_stddev,
                             dtype=tf.float32)
    _image_ = tf.clip_by_value(_image_ + noise, 0.0, 1.0) * 255.0

    return tf.cast(_image_, tf.uint8)


def get_shade_source():
    shade_list = glob.glob('./shade_source/*.png')
    total_shade = []
    for i, shade_name in enumerate(shade_list):
        shade = tf.image.decode_png(tf.read_file(shade_name), channels=1)
        shade = tf.expand_dims(shade, 0)
        shade = tf.image.resize_nearest_neighbor(shade, FLAGS.train_crop_size)
        total_shade.append(shade)
        if i == 100:
            break
    return tf.concat(total_shade, 0)


def add_random_shade(img_src):
    # conditionally switch 0 to 1 and vise versa
    def value_switch(tensor):
        # only works if tensor has only 0 and 1 values
        return tensor - 2 * tensor + 1

    shade_src_total = get_shade_source()
    num_shade = get_shape(shade_src_total)[0]
    rand_num = tf.random_uniform((), maxval=num_shade, dtype=tf.int32)
    shade_src = shade_src_total[rand_num, ::]
    shade_src = tf.cond(tf.equal(tf.random_uniform((), maxval=2, dtype=tf.int32), tf.constant(1)),
                        lambda: value_switch(shade_src), lambda: shade_src)
    img_src = tf.divide(img_src, 255.0)
    row, col, _ = img_src.shape
    shade_src = tf.expand_dims(tf.expand_dims(shade_src, 0), 3)
    shade_src = tf.image.resize_nearest_neighbor(shade_src, [row, col])
    alpha = tf.random_uniform((), minval=0.5, maxval=1.0, dtype=tf.float32)
    shade = tf.multiply(shade_src, alpha)
    shade = tf.squeeze(shade)

    case_true = tf.reshape(tf.multiply(tf.ones(get_shape(tf.reshape(shade, [-1])), tf.float32), 1.0), get_shape(shade))
    case_false = shade
    shade = tf.where(tf.equal(shade, 0), case_true, case_false)
    shade = tf.expand_dims(tf.expand_dims(shade, 0), 3)

    return tf.multiply(img_src, shade)


def _additional_augmentation(_image):
    method_idx = tf.random_uniform((), minval=0, maxval=5, dtype=tf.int32)
    _image = tf.cond(tf.equal(method_idx, tf.constant(0)), lambda: random_brightness(_image), lambda: _image)
    _image = tf.cond(tf.equal(method_idx, tf.constant(1)), lambda: random_contrast(_image), lambda: _image)
    _image = tf.cond(tf.equal(method_idx, tf.constant(2)), lambda: random_hue(_image), lambda: _image)
    _image = tf.cond(tf.equal(method_idx, tf.constant(3)), lambda: random_saturation(_image), lambda: _image)
    _image = tf.cond(tf.equal(method_idx, tf.constant(4)), lambda: add_random_shade(_image), lambda: _image)

    do_gaussian_noise = tf.less_equal(tf.random_uniform([]), FLAGS.prob_add_gaussian_noise)
    return tf.cond(do_gaussian_noise, lambda: add_gaussian_noise(_image), lambda: _image)


def additional_augmentation(image):
    image = tf.cast(image, tf.uint8)
    if FLAGS.additional_augmentation and FLAGS.job == 'train':  # True for add "additional_augmentation" into graph
        do_additional_augmentation = tf.less_equal(tf.random_uniform((), maxval=1.0),
                                                   FLAGS.prob_additional_augmentation)

        return tf.cast(
            tf.cond(do_additional_augmentation, lambda: _additional_augmentation(image), lambda: image),
            tf.float32)
    else:
        return tf.cast(image, tf.float32)
