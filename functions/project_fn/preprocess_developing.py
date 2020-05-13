from functions.project_fn.misc_utils import get_tensor_shape
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import cv2 as cv
import numpy as np


class Preprocessing:
    def __init__(self, config):
        self.config = config

    def __getattr__(self, item):
        try:
            return getattr(self.config, item)
        except AttributeError:
            raise AttributeError("'config' has no attribute '%s'" % item)

    def _get_random_scale(self):
        if self.min_random_scale_factor < 0:
            raise ValueError("min_scale_factor cannot be nagative value")
        if self.min_random_scale_factor > self.max_random_scale_factor:
            raise ValueError("min_scale_factor must be larger than max_scale_factor")
        if self.max_random_scale_factor == 1 and self.max_random_scale_factor == self.min_random_scale_factor:
            return 1.0
        elif self.min_random_scale_factor == self.max_random_scale_factor:
            return tf.cast(self.min_random_scale_factor, tf.float32)
        else:
            return tf.random_uniform([], minval=self.min_random_scale_factor, maxval=self.max_random_scale_factor)

    def _randomly_scale_image_and_label(self, image, label):
        """Randomly scales image and label.

        Args:
          image: Image with original_shape [height, width, 3].
          label: Label with original_shape [height, width, 1].
          scale: The value to scale image and label.

        Returns:
          Scaled image and label.
        """
        scale = self._get_random_scale()
        image_shape = get_tensor_shape(image)
        new_dim = tf.cast(tf.cast([image_shape[0], image_shape[1]], tf.float32) * scale, tf.int32)

        # Need squeeze and expand_dims because image interpolation takes
        # 4D tensors as input.
        image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), new_dim, align_corners=True), [0])
        label = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_dim, align_corners=True), [0])
        return image, label

    def _random_crop(self, image_list):
        """Crops the given list of images.

        The function applies the same crop to each img in the list. This can be
        effectively applied when there are multiple img inputs of the same
        dimension such as:

          img, depths, normals = random_crop([img, depths, normals], 120, 150)

        Args:
          image_list: a list of img tensors of the same dimension but possibly
            varying channel.

        Returns:
          the sharp_image_list with cropped images.

        Raises:
          ValueError: if there are multiple img inputs provided with different size
            or the images are smaller than the crop dimensions.
        """

        def _crop(img, offset_h, offset_w, crop_height, crop_width):
            """Crops the given image using the provided offsets and sizes.

            Note that the method doesn"t assume we know the input image size but it does
            assume we know the input image rank.

            Args:
              img: an image of original_shape [height, width, channels].
              offset_h: a scalar tensor indicating the height offset.
              offset_w: a scalar tensor indicating the width offset.
              crop_height: the height of the cropped image.
              crop_width: the width of the cropped image.

            Returns:
              The cropped (and resized) image.

            Raises:
              ValueError: if `image` doesn"t have rank of 3.
              InvalidArgumentError: if the rank is not 3 or if the image dimensions are
                less than the crop size.
            """
            original_shape = tf.shape(img)

            if len(img.get_shape().as_list()) != 3:
                raise ValueError("input must have rank of 3")
            original_channels = img.get_shape().as_list()[2]

            rank_assertion = tf.Assert(
                tf.equal(tf.rank(img), 3),
                ["Rank of img must be equal to 3."])
            with tf.control_dependencies([rank_assertion]):
                cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

            size_assertion = tf.Assert(
                tf.logical_and(
                    tf.greater_equal(original_shape[0], crop_height),
                    tf.greater_equal(original_shape[1], crop_width)),
                ["Crop size greater than the img size."])

            offsets = tf.cast(tf.stack([offset_h, offset_w, 0]), tf.int32)

            # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
            # define the crop size.
            with tf.control_dependencies([size_assertion]):
                img = tf.slice(img, offsets, cropped_shape)
            img = tf.reshape(img, cropped_shape)
            img.set_shape([crop_height, crop_width, original_channels])
            return img

        if not image_list:
            raise ValueError("Empty sharp_image_list.")
        image_list = [img for img in image_list if img is not None]

        # Compute the rank assertions.
        rank_assertions = []
        for i in range(len(image_list)):
            image_rank = tf.rank(image_list[i])
            rank_assert = tf.Assert(tf.equal(image_rank, 3),
                                    ["Wrong rank for tensor %s [expected] [actual]",
                                     image_list[i].name, 3, image_rank])
            rank_assertions.append(rank_assert)

        with tf.control_dependencies([rank_assertions[0]]):
            image_shape = get_tensor_shape(image_list[0])
        image_height = image_shape[0]
        image_width = image_shape[1]
        crop_size_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(image_height, self.crop_size[0]),
                           tf.greater_equal(image_width, self.crop_size[1])),
            ["Crop size greater than the img size."])

        asserts = [rank_assertions[0], crop_size_assert]

        for i in range(1, len(image_list)):
            image = image_list[i]
            asserts.append(rank_assertions[i])
            with tf.control_dependencies([rank_assertions[i]]):
                shape = get_tensor_shape(image)
            height = shape[0]
            width = shape[1]

            height_assert = tf.Assert(
                tf.equal(height, image_height),
                ["Wrong height for tensor %s [expected][actual]",
                 image.name, height, image_height])
            width_assert = tf.Assert(
                tf.equal(width, image_width),
                ["Wrong width for tensor %s [expected][actual]",
                 image.name, width, image_width])
            asserts.extend([height_assert, width_assert])

        # Create a random bounding box.
        #
        # Use tf.random_uniform and not numpy.random.rand as doing the former would
        # generate random numbers at graph eval time, unlike the latter which
        # generates random numbers at graph definition time.
        with tf.control_dependencies(asserts):
            max_offset_height = tf.reshape(image_height - self.crop_size[0] + 1, [])
            max_offset_width = tf.reshape(image_width - self.crop_size[1] + 1, [])
        offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
        offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

        out_list = [_crop(image, offset_height, offset_width, self.crop_size[0], self.crop_size[1]) for image in image_list]
        if len(out_list) == 2:
            return out_list[0], out_list[1]
        elif len(out_list) == 1:
            return out_list[0], None
        else:
            raise ValueError("Unexpected out_list. This function is designed for a list of two image tensors (either rank 3 or rank1)")

    def _flip(self, tensor_list_in):
        tensor_list_out = []
        if self.flip_probability > 0:
            flip_prob = tf.constant(self.flip_probability)
            is_flipped = tf.less_equal(tf.random_uniform([]), flip_prob)
            for tensor in tensor_list_in:
                if tensor is not None:
                    tensor_list_out.append(tf.cond(is_flipped, lambda: tf.image.flip_left_right(tensor), lambda: tensor))
                else:
                    tensor_list_out.append(None)
            return tensor_list_out
        else:
            return tensor_list_in

    def _rotate(self, tensor_list_in):
        tensor_list_out = []
        if self.rotate_probability > 0:
            rotate_prob = tf.constant(self.rotate_probability)
            is_rotate = tf.less_equal(tf.random_uniform([]), rotate_prob)
            if self.rotate_angle_by90:
                rotate_k = tf.random_uniform((), maxval=4, dtype=tf.int32)
                for tensor in tensor_list_in:
                    if tensor is not None:
                        tensor_list_out.append(tf.cond(is_rotate, lambda: tf.image.rot90(tensor, rotate_k), lambda: tensor))
                    else:
                        tensor_list_out.append(None)
            else:
                angle = tf.random.uniform((), minval=self.rotate_angle_range[0], maxval=self.rotate_angle_range[1], dtype=tf.float32)
                for tensor in tensor_list_in:
                    if tensor is not None:
                        tensor_list_out.append(tf.cond(is_rotate, lambda: tf.contrib.image.rotate(tensor, angle, interpolation="BILINEAR"), lambda: tensor))
                    else:
                        tensor_list_out.append(None)

            return tensor_list_out
        else:
            return tensor_list_in

    def add_gaussian_noise(self, tensor, stddev_min=0.01, stddev_max=0.09):
        stddev = tf.random.uniform([], minval=stddev_min, maxval=stddev_max, dtype=tf.float32)
        tensor = tf.to_float(tensor) / 255.0
        noise = tf.random_normal(shape=tf.shape(tensor), mean=0.0, stddev=stddev, dtype=tf.float32)
        return tf.clip_by_value(tensor + noise, 0.0, 1.0) * 255.0

    def rgb_perturb(self, tensor):
        with tf.device("/device:CPU:0"):
            tensor = tf.transpose(tensor, [3, 1, 2, 0])
            tensor = tf.random.shuffle(tensor)
            return tf.transpose(tensor, [3, 1, 2, 0])

    def random_quality(self, tensor):
        tensor = tf.squeeze(tensor)
        with tf.device("/device:CPU:0"):
            return tf.cast(tf.image.random_jpeg_quality(tf.cast(tensor, tf.uint8), 30, 50), tf.float32)[tf.newaxis, ::]

    def normalize_input(self, input_tensor, scale=1.3):
        # set pixel values from 0 to 1
        # return tf.cast(input_tensor, tf.float32) / 255.0
        if scale != 1.0:
            return (tf.cast(input_tensor, input_tensor.dtype) / 127.5 - 1) * scale
        else:
            return tf.cast(input_tensor, input_tensor.dtype) / 127.5 - 1

    def normalize_input2(self, input_tensor):
        # set pixel values from 0 to 1
        # return tf.cast(input_tensor, tf.float32) / 255.0
        return (tf.cast(input_tensor, input_tensor.dtype) / 127.5 - 1) * 1.0

    def normalize_input3(self, input_tensor):
        b, h, w, c = get_tensor_shape(input_tensor)
        mean = [0.485, 0.456, 0.406]
        mean = np.expand_dims(np.expand_dims(mean, 0), 0)
        mean = tf.constant(np.stack([mean] * b, 0), tf.float32)
        std = [0.229, 0.224, 0.225]
        std = np.expand_dims(np.expand_dims(std, 0), 0)
        std = tf.constant(np.stack([std] * b, 0), tf.float32)
        normalized = (input_tensor / 255.0 - mean) / std
        return normalized

    @staticmethod
    def warp(img, gt, prob, ratio, warp_crop_prob):
        if np.random.rand() <= prob:
            def rnd(length):
                return np.random.randint(0, int(length * ratio))

            h, w, _ = img.shape
            # scale up 3 times just in case seg has very thin line of labels
            img = cv.resize(img, (w * 4, h * 4))
            gt = cv.resize(gt, (w * 4, h * 4))
            new_h, new_w, _ = img.shape

            pts1 = np.float32([[0, 0], [new_w, 0], [0, new_h], [new_w, new_h]])  # [width, height]
            pts2 = np.float32([[rnd(new_w), rnd(new_h)], [new_w - rnd(new_w), rnd(new_h)], [rnd(new_w), new_h - rnd(new_h)], [new_w - rnd(new_w), new_h - rnd(new_h)]])

            matrix = cv.getPerspectiveTransform(pts1, pts2)

            warped_img = cv.warpPerspective(img, matrix, (new_w, new_h), flags=cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)
            warped_gt = cv.warpPerspective(gt, matrix, (new_w, new_h), flags=cv.INTER_NEAREST + cv.WARP_FILL_OUTLIERS)

            if np.random.rand() <= warp_crop_prob:
                w1 = int(max(pts2[0][0], pts2[2][0]))
                w2 = int(min(pts2[1][0], pts2[3][0]))

                h1 = int(max(pts2[0][1], pts2[1][1]))
                h2 = int(min(pts2[2][1], pts2[3][1]))

                warped_img = warped_img[h1:h2, w1:w2, :]
                warped_gt = warped_gt[h1:h2, w1:w2]
            warped_img = cv.resize(warped_img, (w, h))
            warped_gt = cv.resize(warped_gt, (w, h))
            return warped_img, warped_gt
        return img, gt

    def additional_augmentation(self, image, config):
        image = tf.cast(image, tf.uint8)
        h, w, c = get_tensor_shape(image)

        def _random_quality(_image_):
            _image_ = tf.image.random_jpeg_quality(_image_, config.random_quality[0], config.random_quality[1])
            _image_.set_shape([h, w, c])
            return _image_

        def _rgb_permutation(_image_):
            tensor = tf.transpose(_image_, [2, 0, 1])
            tensor = tf.random.shuffle(tensor)
            return tf.transpose(tensor, [1, 2, 0])

        def _random_brightness(_image_):
            return tf.image.adjust_brightness(_image_, tf.random_uniform([], maxval=config.brightness_constant))

        def _random_contrast(_image_):
            return tf.image.adjust_contrast(_image_, tf.random_uniform([],
                                                                       minval=config.contrast_constant[0],
                                                                       maxval=config.contrast_constant[1]))

        def _random_hue(_image_):
            return tf.image.adjust_hue(_image_, tf.random_uniform([],
                                                                  minval=config.hue_constant[0],
                                                                  maxval=config.hue_constant[1]))

        def _random_saturation(_image_):
            return tf.image.adjust_saturation(_image_, tf.random_uniform([],
                                                                         minval=config.saturation_constant[0],
                                                                         maxval=config.saturation_constant[1]))

        def _random_gaussian_noise(_image_):
            _image_ = tf.cast(_image_, tf.float32) / 255.0
            rnd_stddev = tf.random_uniform([], minval=config.gaussian_noise_std[0], maxval=config.gaussian_noise_std[1])
            noise = tf.random_normal(shape=tf.shape(_image_), mean=0.0, stddev=rnd_stddev, dtype=tf.float32)
            _image_ = tf.clip_by_value(_image_ + noise, 0.0, 1.0) * 255.0
            return tf.cast(_image_, tf.uint8)

        def _shred(_image_, split_axis, shred_num, shift_ratio):
            _image_shape = get_tensor_shape(_image_)
            split_indices = np.linspace(0, _image_shape[split_axis], shred_num + 1, dtype=np.int32)
            split_indices = split_indices[1:] - split_indices[:-1]

            splitted = tf.split(_image_, split_indices, split_axis)
            pad_size = int(_image_shape[split_axis] * shift_ratio)
            padded_container = []
            for stip in splitted:
                rnd0 = tf.random.uniform((), maxval=2, dtype=tf.int32)
                rnd1 = 1 - rnd0
                range1 = rnd0 * pad_size
                range2 = rnd1 * pad_size
                pad = tf.cond(tf.equal(split_axis, 0), lambda: [[0, 0], [range1, range2], [0, 0]], lambda: [[range1, range2], [0, 0], [0, 0]])
                padded_container.append(tf.pad(stip, pad, "REFLECT"))
            shreded = tf.concat(padded_container, split_axis)
            range1 = int(pad_size * 0.5)
            range2 = pad_size - range1
            shreded = tf.cond(tf.equal(split_axis, 0), lambda: shreded[:, range1:-range2, :], lambda: shreded[range1:-range2, ::])
            # _image_shape[1 - split_axis] = _image_shape[1 - split_axis] - pad_size
            shreded.set_shape(_image_shape)
            return shreded

        def elastic_transform(_image, alpha, sigma):
            """Elastic deformation of images as described in [Simard2003]_.
            .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
               Convolutional Neural Networks applied to Visual Document Analysis", in
               Proc. of the International Conference on Document Analysis and
               Recognition, 2003.
            """
            random_state = np.random.RandomState(None)

            shape = _image.shape
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

            distored_image = map_coordinates(_image, indices, order=1, mode="reflect")
            return distored_image.reshape(_image.shape)

        def _random_shade(_image_):
            decode_features = {"shade": tf.FixedLenFeature((), tf.string, default_value=""),
                               "height": tf.FixedLenFeature([], tf.int64),
                               "width": tf.FixedLenFeature([], tf.int64)}

            def shade_getter(tfrecord):
                def parser(data):
                    parsed = tf.parse_single_example(data, decode_features)
                    # h = tf.convert_to_tensor(parsed["height"])
                    # w = tf.convert_to_tensor(parsed["width"])
                    return tf.convert_to_tensor(tf.image.decode_png(parsed["shade"], channels=1))

                dataset = tf.data.TFRecordDataset(tfrecord).repeat()
                return dataset.apply(tf.data.experimental.map_and_batch(parser, 1)).make_one_shot_iterator().get_next()

            def reverse_value(_shade_src_):
                # only works if tensor has only 0 and 1 values
                return _shade_src_ - 2 * _shade_src_ + 1

            shade_src = tf.cast(shade_getter(config.shade_source), tf.float32)

            shade_src = tf.cond(tf.equal(tf.random_uniform((), maxval=2, dtype=tf.int32), tf.constant(1)),
                                lambda: reverse_value(shade_src), lambda: shade_src)
            shade_shape = get_tensor_shape(shade_src)
            image_shape = get_tensor_shape(_image_)
            min_shade_length = tf.cast(tf.reduce_min(shade_shape[1:3]), tf.float32)
            max_image_length = tf.cast(tf.reduce_max(image_shape), tf.float32)

            def rescale_shade(_shade_src_):
                scale_factor = max_image_length / min_shade_length
                # _shade_src_ = tf.expand_dims(_shade_src_, 0)
                # random scale up of shade_src in within the range of 1.0 to 2.0
                rnd_factor = tf.random_uniform([], minval=1.0, maxval=2.0)
                target_h = tf.cast(tf.cast(shade_shape[1], tf.float32) * scale_factor * rnd_factor, tf.int32)
                target_w = tf.cast(tf.cast(shade_shape[2], tf.float32) * scale_factor * rnd_factor, tf.int32)
                # target_c = shade_shape[3]
                return tf.image.resize_nearest_neighbor(_shade_src_, [target_h, target_w], align_corners=True)
                # return tf.reshape(_shade_src, [target_h, target_w, target_c])

            # the min length of shade_src will be at least equal to the max length of img
            shade_src = tf.cond(tf.not_equal(max_image_length, min_shade_length), lambda: rescale_shade(shade_src), lambda: shade_src)
            shade_shape = get_tensor_shape(shade_src)
            shade_src = tf.reshape(shade_src, shade_shape[1::])

            # random crop
            shade_src, _ = self._random_crop([shade_src, None])

            # check if shade_src is greater than img
            alpha = tf.random_uniform((), minval=0.5, maxval=1.0, dtype=tf.float32)
            shade = shade_src * alpha

            case_true = tf.reshape(tf.multiply(tf.ones(get_tensor_shape(tf.reshape(shade, [-1])), tf.float32), 1.0), get_tensor_shape(shade))
            case_false = shade
            shade = tf.where(tf.equal(shade, 0), case_true, case_false)
            # img_normalized = _image_ / 255.0
            return tf.cast(tf.multiply(tf.cast(_image_, tf.float32), shade), tf.uint8)

        def draw_grid(im, grid_size):
            # Draw grid lines
            for i in range(0, im.shape[1], grid_size):
                cv.line(im, (i, 0), (i, im.shape[0]), color=(255,))
            for j in range(0, im.shape[0], grid_size):
                cv.line(im, (0, j), (im.shape[1], j), color=(255,))
            return im

        def _additional_augmentation(_image):
            if config.random_quality_prob > 0.0:
                do_quality = tf.less_equal(tf.random_uniform([]), config.random_quality_prob)
                _image = tf.cond(do_quality, lambda: _random_quality(_image), lambda: _image)

            if config.rgb_permutation_prob > 0.0:
                do_permutation = tf.less_equal(tf.random_uniform([]), config.rgb_permutation_prob)
                _image = tf.cond(do_permutation, lambda: _rgb_permutation(_image), lambda: _image)

            if config.brightness_prob > 0.0:
                do_brightness = tf.less_equal(tf.random_uniform([]), config.brightness_prob)
                _image = tf.cond(do_brightness, lambda: _random_brightness(_image), lambda: _image)

            if config.contrast_prob > 0.0:
                do_contrast = tf.less_equal(tf.random_uniform([]), config.contrast_prob)
                _image = tf.cond(do_contrast, lambda: _random_contrast(_image), lambda: _image)

            if config.hue_prob > 0.0:
                do_hue = tf.less_equal(tf.random_uniform([]), config.hue_prob)
                _image = tf.cond(do_hue, lambda: _random_hue(_image), lambda: _image)

            if config.saturation_prob > 0.0:
                do_saturation = tf.less_equal(tf.random_uniform([]), config.saturation_prob)
                _image = tf.cond(do_saturation, lambda: _random_saturation(_image), lambda: _image)

            if config.gaussian_noise_prob > 0.0:
                do_gaussian_noise = tf.less_equal(tf.random_uniform([]), config.gaussian_noise_prob)
                _image = tf.cond(do_gaussian_noise, lambda: _random_gaussian_noise(_image), lambda: _image)

            if config.shred_vertical_prob > 0.0:
                shred_num = config.shred_num
                shift_ratio = config.shift_ratio
                do_shred = tf.less_equal(tf.random_uniform([]), config.shred_vertical_prob)
                _image = tf.cond(do_shred, lambda: _shred(_image, 0, shred_num, shift_ratio), lambda: _image)
            if config.shred_horizontal_prob > 0.0:
                shred_num = config.shred_num
                shift_ratio = config.shift_ratio
                do_shred = tf.less_equal(tf.random_uniform([]), config.shred_horizontal_prob)
                _image = tf.cond(do_shred, lambda: _shred(_image, 1, shred_num, shift_ratio), lambda: _image)

            if config.elastic_distortion_prob > 0.0:
                do_elastic_transform = tf.less_equal(tf.random_uniform([]), config.elastic_distortion_prob)
                # _image = tf.py_func(draw_grid, [_image, 5], tf.uint8)
                _image = tf.cond(do_elastic_transform, lambda: tf.py_func(elastic_transform, [_image, 691, 6], tf.uint8), lambda: _image)

            if config.shade_prob > 0.0:
                do_shade = tf.less_equal(tf.random_uniform([]), config.shade_prob)
                _image = tf.cond(do_shade, lambda: _random_shade(_image), lambda: _image)
            return _image

        do_additional_augmentation = tf.less_equal(tf.random_uniform((), maxval=1.0), config.additional_augmentation_probability)
        return tf.cast(tf.cond(do_additional_augmentation, lambda: _additional_augmentation(image), lambda: image), config.dtype)

    def preprocessing(self, img, gt):
        img = tf.cast(img, self.dtype)
        if self.is_train:
            if gt is None:
                raise ValueError("gt should not be none in training")

            # Data augmentation by randomly scaling the inputs.
            if self.min_random_scale_factor != 1.0 or self.max_random_scale_factor != 1.0:
                if self.min_random_scale_factor > self.max_random_scale_factor:
                    raise ValueError("min_scale_factor should be smaller than max_scale factor")
                img, gt = self._randomly_scale_image_and_label(img, gt)

            # Randomly crop the img and gt.
            img, gt = self._random_crop([img, gt])
            img.set_shape([self.crop_size[0], self.crop_size[1], None])
            gt.set_shape([self.crop_size[0], self.crop_size[1], 1])
            img, gt = self._flip([img, gt])
            img, gt = self._rotate([img, gt])
            if self.warp_prob > 0.0:
                h, w, _ = get_tensor_shape(gt)
                gt = tf.squeeze(gt)
                img = tf.cast(img, tf.uint8)
                gt = tf.cast(gt, tf.uint8)
                img, gt = tf.py_func(self.warp, [img, gt, self.warp_prob, self.warp_ratio, self.warp_crop_prob], [tf.uint8, tf.uint8])
                # img, gt = tf.cond(do_warp, lambda: tf.py_func(warp, [img, gt, config], [tf.float32, tf.uint8]), lambda: (img, gt))
                img = tf.cast(img, self.dtype)
                img.set_shape([h, w, 3])
                gt.set_shape([h, w])
                gt = gt[:, :, tf.newaxis]
            if self.additional_augmentation_probability > 0.0:
                img = additional_augmentation(img, config)
        return img, gt
