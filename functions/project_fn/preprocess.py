from functions.project_fn.utils import get_tensor_shape as get_shape
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import cv2 as cv
import numpy as np


class Preprocessing:
    @staticmethod
    def _fp32(tensor_or_list):
        """
        tensor: either a tensor or a list of tensors
        """
        if tf.is_tensor(tensor_or_list):
            return tf.cast(tensor_or_list, tf.float32)
        elif isinstance(tensor_or_list, list):
            out_list = []
            for tensor in tensor_or_list:
                out_list.append(tf.cast(tensor, tf.float32))
            return out_list

    @staticmethod
    def _uint8(tensor_or_list):
        """
        tensor: either a tensor or a list of tensors
        """
        if tf.is_tensor(tensor_or_list):
            return tf.cast(tensor_or_list, tf.uint8)
        elif isinstance(tensor_or_list, list):
            out_list = []
            for tensor in tensor_or_list:
                out_list.append(tf.cast(tensor, tf.uint8))
            return out_list

    @staticmethod
    def draw_grid(im, grid_num):
        """
        draw_grid lines for visualizing how an image is manipuliated in data augmentation

        """
        grid_size = int(im.shape[1] / grid_num)
        for i in range(0, im.shape[1], grid_size):
            cv.line(im, (i, 0), (i, im.shape[0]), color=(255,))
        for j in range(0, im.shape[0], grid_size):
            cv.line(im, (0, j), (im.shape[1], j), color=(255,))
        return im

    @staticmethod
    def _warp(image, gt, prob, ratio, warp_crop_prob):
        """
        Change perspective.
        The function

        """
        if not 0.0 < ratio <= 1.0:
            raise ValueError("warp ratio should be (0.0, 1.0]")

        if np.random.rand() <= prob:
            def rnd(length):
                return np.random.randint(0, int(length * ratio))

            h, w, _ = image.shape
            # scale up 4 times just in case seg has very thin line of labels
            image = cv.resize(image, (w * 4, h * 4))
            gt = cv.resize(gt, (w * 4, h * 4))
            new_h, new_w, _ = image.shape

            pts1 = np.float32([[0, 0], [new_w, 0], [0, new_h], [new_w, new_h]])  # [width, height]
            pts2 = np.float32([[rnd(new_w), rnd(new_h)], [new_w - rnd(new_w), rnd(new_h)], [rnd(new_w), new_h - rnd(new_h)], [new_w - rnd(new_w), new_h - rnd(new_h)]])

            matrix = cv.getPerspectiveTransform(pts1, pts2)

            warped_image = cv.warpPerspective(image, matrix, (new_w, new_h), flags=cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)
            warped_gt = cv.warpPerspective(gt, matrix, (new_w, new_h), flags=cv.INTER_NEAREST + cv.WARP_FILL_OUTLIERS)

            if np.random.rand() <= warp_crop_prob:
                w1 = int(max(pts2[0][0], pts2[2][0]))
                w2 = int(min(pts2[1][0], pts2[3][0]))

                h1 = int(max(pts2[0][1], pts2[1][1]))
                h2 = int(min(pts2[2][1], pts2[3][1]))

                warped_image = warped_image[h1:h2, w1:w2, :]
                warped_gt = warped_gt[h1:h2, w1:w2]
            warped_image = cv.resize(warped_image, (w, h))
            warped_gt = cv.resize(warped_gt, (w, h))
            return warped_image, warped_gt
        return image, gt

    @staticmethod
    def elastic_transform(img_gt_pair, prob):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        if np.random.rand() <= prob:
            random_state = np.random.RandomState(None)
            # scale up 4 times just in case seg has very thin line of labels
            shape = img_gt_pair.shape
            shape_size = shape[:2]

            # (image, alpha, sigma, alpha_affine, random_state=None)
            alpha = shape[1] * np.random.randint(1, 4)
            sigma = shape[1] * np.random.uniform(0.05, 0.08)
            alpha_affine = shape[1] * np.random.uniform(0.05, 0.08)

            # Random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size], center_square - square_size])
            pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
            affinie_matrix = cv.getAffineTransform(pts1, pts2)
            img_gt_pair = cv.warpAffine(img_gt_pair, affinie_matrix, shape_size[::-1], borderMode=cv.BORDER_REFLECT_101, flags=cv.INTER_AREA)

            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
            return map_coordinates(img_gt_pair, indices, order=1, mode='reflect').reshape(shape)

    def _get_random_scale(self):
        if self.random_scale_range[0] < 0:
            raise ValueError("min_scale_factor cannot be nagative value")
        if self.random_scale_range[0] > self.random_scale_range[1]:
            raise ValueError("min_scale_factor must be larger than max_scale_factor")
        elif self.random_scale_range[0] == self.random_scale_range[0]:
            return tf.cast(self.random_scale_range[0], tf.float32)
        else:
            return tf.random_uniform([], minval=self.random_scale_range[0], maxval=self.random_scale_range[0])

    def _randomly_scale_image_and_label(self):
        """Randomly scales image and label.

        Args:
          image: Image with original_shape [height, width, 3].
          label: Label with original_shape [height, width, 1].
          scale: The value to scale image and label.

        Returns:
          Scaled image and label.
        """
        scale = self._get_random_scale()
        h, w, c = get_shape(self.image)
        new_dim = tf.cast(tf.cast([h, w], tf.float32) * scale, tf.int32)

        # Need squeeze and expand_dims because image interpolation takes
        # 4D tensors as input.
        self.image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(self.image, 0), new_dim, align_corners=True), [0])
        self.gt = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(self.gt, 0), new_dim, align_corners=True), [0])

    def _random_crop(self):
        # concat in channel
        image_gt_pair = tf.concat([self.image, self.gt], 2)
        image_gt_pair_cropped = tf.image.random_crop(image_gt_pair, [self.crop_size[0], self.crop_size[1], 4])
        self.image = image_gt_pair_cropped[:, :, :3]
        self.gt = image_gt_pair_cropped[:, :, 3:]

    def _flip(self):
        if self.flip_probability > 0:
            do_flip = tf.less_equal(tf.random_uniform([]), self.flip_probability)
            self.image = tf.cond(do_flip, lambda: tf.image.flip_left_right(self.image), lambda: self.image)
            self.gt = tf.cond(do_flip, lambda: tf.image.flip_left_right(self.gt), lambda: self.gt)

    def _rotate(self):
        if self.rotate_probability > 0:
            on_off = tf.less_equal(tf.random_uniform([]), self.rotate_probability)
            if self.rotate_angle_by90:
                rotate_k = tf.random_uniform((), maxval=3, dtype=tf.int32)
                self.image = tf.cond(on_off, lambda: tf.image.rot90(self.image, rotate_k), lambda: self.image)
                self.gt = tf.cond(on_off, lambda: tf.image.rot90(self.gt, rotate_k), lambda: self.gt)
            else:
                angle = tf.random.uniform((), minval=self.rotate_angle_range[0], maxval=self.rotate_angle_range[1], dtype=tf.float32)
                self.image = tf.cond(on_off, lambda: tf.contrib.image.rotate(self.image, angle, interpolation="BILINEAR"))
                self.gt = tf.cond(on_off, lambda: tf.contrib.image.rotate(self.gt, angle, interpolation="NEAREST"))

    def _random_quality(self):
        do_quality = tf.less_equal(tf.random_uniform([]), self.random_quality_prob)
        self.image = tf.cond(do_quality,
                             lambda: tf.image.random_jpeg_quality(self.image, self.random_quality[0], self.random_quality[1]),
                             lambda: self.image)
        self.image.set_shape([self.crop_size[0], self.crop_size[1], 3])

    def _rgb_permutation(self):
        def execute_fn(image):
            image = tf.transpose(image, [2, 0, 1])
            image = tf.random.shuffle(image)
            return tf.transpose(image, [1, 2, 0])

        do_permutation = tf.less_equal(tf.random_uniform([]), self.rgb_permutation_prob)
        self.image = tf.cond(do_permutation, lambda: execute_fn(self.image), lambda: self.image)

    def _random_brightness(self):
        do_brightness = tf.less_equal(tf.random_uniform([]), self.brightness_prob)
        delta = tf.random_uniform([], maxval=self.brightness_constant)
        self.image = tf.cond(do_brightness,
                             lambda: tf.image.adjust_brightness(self.image, delta),
                             lambda: self.image)

    def _random_contrast(self):
        do_contrast = tf.less_equal(tf.random_uniform([]), self.contrast_prob)
        contrast_factor = tf.random_uniform([], minval=self.contrast_constant[0], maxval=self.contrast_constant[1])
        self.image = tf.cond(do_contrast,
                             lambda: tf.image.adjust_contrast(self.image, contrast_factor),
                             lambda: self.image)

    def _random_hue(self):
        do_hue = tf.less_equal(tf.random_uniform([]), self.hue_prob)
        delta = tf.random_uniform([], minval=self.hue_constant[0], maxval=self.hue_constant[1])
        self.image = tf.cond(do_hue,
                             lambda: tf.image.adjust_hue(self.image, delta),
                             lambda: self.image)

    def _random_saturation(self):
        do_saturation = tf.less_equal(tf.random_uniform([]), self.saturation_prob)
        saturation_factor = tf.random_uniform([], minval=self.saturation_constant[0], maxval=self.saturation_constant[1])
        self.image = tf.cond(do_saturation,
                             lambda: tf.image.adjust_saturation(self.image, saturation_factor),
                             lambda: self.image)

    def _random_gaussian_noise(self):
        def execute_fn(image, std):
            image = image / 255.0
            rnd_stddev = tf.random_uniform([], minval=std[0], maxval=std[1])
            noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=rnd_stddev, dtype=tf.float32)
            return tf.clip_by_value(image + noise, 0.0, 1.0) * 255.0

        do_gaussian_noise = tf.less_equal(tf.random_uniform([]), self.gaussian_noise_prob)
        self.image = tf.cond(do_gaussian_noise,
                             lambda: execute_fn(self.image, self.gaussian_noise_std),
                             lambda: self.image)

    def _random_shred(self):
        # todo: make this function work for tensors
        def execute_fn(image, gt, shred_range):
            image_shape = get_shape(image)
            gt_shape = get_shape(gt)
            shred_num = tf.random.uniform([], minval=shred_range[0], maxval=shred_range[1] + 1, dtype=tf.uint8)
            for split_axis in [0, 1]:
                split_indices = np.linspace(0, image_shape[split_axis], shred_num + 1, dtype=np.int32)
                # tf.linspace(0, image_shape[split_axis], shred_num+1)
                split_indices = split_indices[1:] - split_indices[:-1]
                splitted_image = tf.split(self.image, split_indices, split_axis)
                splitted_gt = tf.split(self.gt, split_indices, split_axis)
                pad_size = int(image_shape[split_axis] * self.shred_shift_ratio)
                padded_image_container = []
                padded_gt_container = []
                for strip_image, strip_gt in zip(splitted_image, splitted_gt):
                    rnd0 = tf.random.uniform((), maxval=2, dtype=tf.int32)
                    rnd1 = 1 - rnd0
                    range1 = rnd0 * pad_size
                    range2 = rnd1 * pad_size
                    pad = tf.cond(tf.equal(split_axis, 0), lambda: [[0, 0], [range1, range2], [0, 0]], lambda: [[range1, range2], [0, 0], [0, 0]])
                    padded_image_container.append(tf.pad(strip_image, pad, "REFLECT"))
                    padded_gt_container.append(tf.pad(strip_gt, pad, "REFLECT"))
                shredded_image = tf.concat(padded_image_container, split_axis)
                shredded_gt = tf.concat(padded_gt_container, split_axis)
                range1 = int(pad_size * 0.5)
                range2 = pad_size - range1
                shredded_image = tf.cond(tf.equal(split_axis, 0), lambda: shredded_image[:, range1:-range2, :], lambda: shredded_image[range1:-range2, ::])
                shredded_gt = tf.cond(tf.equal(split_axis, 0), lambda: shredded_gt[:, range1:-range2, :], lambda: shredded_gt[range1:-range2, ::])
                shredded_image.set_shape(image_shape)
                shredded_gt.set_shape(gt_shape)
                return shredded_image, shredded_gt

        do_shred = tf.less_equal(tf.random_uniform([]), self.shred_prob)
        self.image, self.gt = tf.cond(do_shred,
                                      lambda: execute_fn(self.image, self.gt, self.shred_piece_range),
                                      lambda: self.image)

    def _random_shade(self):
        # _build_input_pipeline shade pipeline
        shade_tfrecord_feature = {"shade": tf.FixedLenFeature((), tf.string, default_value=""),
                                  "height": tf.FixedLenFeature([], tf.int64),
                                  "width": tf.FixedLenFeature([], tf.int64)}

        def shade_pipeline(tfrecord):
            def shade_parser(data):
                parsed = tf.parse_single_example(data, shade_tfrecord_feature)
                return tf.convert_to_tensor(tf.image.decode_png(parsed["shade"], channels=1))

            data = tf.data.TFRecordDataset(tfrecord).repeat()
            data = data.apply(tf.data.experimental.map_and_batch(shade_parser, 1))
            data = data.make_one_shot_iterator().get_next()
            return tf.cast(data, tf.float32)

        def execute_fn(shade_src, image):
            shade_n, shade_h, shade_w, shade_c = get_shape(shade_src)
            image_h, image_w, image_c = get_shape(image)
            min_shade_length = tf.cast(tf.reduce_min([shade_h, shade_w]), tf.float32)
            max_image_length = tf.cast(tf.reduce_max([image_h, image_w]), tf.float32)

            def reverse_value(shade_source):
                return shade_source * -1 + 1

            shade_src = tf.cond(tf.equal(tf.random_uniform((), maxval=2, dtype=tf.int32), tf.constant(1)),
                                lambda: reverse_value(shade_src),
                                lambda: shade_src)

            scale_factor = max_image_length / min_shade_length
            rnd_modifier = tf.random_uniform([], minval=1.0, maxval=2.0)
            shade_h = tf.cast(tf.cast(shade_h, tf.float32) * scale_factor * rnd_modifier, tf.int32)
            shade_w = tf.cast(tf.cast(shade_w, tf.float32) * scale_factor * rnd_modifier, tf.int32)
            shade_src = tf.cond(tf.not_equal(max_image_length, min_shade_length),
                                lambda: tf.image.resize_nearest_neighbor(shade_src, [shade_h, shade_w], align_corners=True),
                                lambda: shade_src)  # now shade is always bigger than image size
            # random crop
            shade_src = tf.squeeze(shade_src, axis=0)
            shade_src = tf.random_crop(shade_src, [image_h, image_w, 1])
            alpha = tf.random_uniform((), minval=0.3, maxval=1.0, dtype=tf.float32)
            shade = shade_src * alpha
            case_true = tf.reshape(tf.multiply(tf.ones(get_shape(tf.reshape(shade, [-1])), tf.float32), 1.0), get_shape(shade))
            case_false = shade
            shade = tf.where(tf.equal(shade, 0), case_true, case_false)
            return tf.multiply(tf.cast(image, tf.float32), shade)

        shade_src = shade_pipeline(self.shade_file)
        do_shade = tf.less_equal(tf.random_uniform([]), self.shade_prob)
        self.image = tf.cond(do_shade, lambda: execute_fn(shade_src, self.image), lambda: self.image)

    def preprocessing(self):
        if self.is_train:
            if self.image is None:
                raise ValueError("image should not be none")
            if self.gt is None:
                raise ValueError("gt should not be none in training")

                # Data augmentation by randomly scaling the inputs.
            if self.random_scale_range != [1.0, 1.0] and self.random_scale_range is not None:
                self._randomly_scale_image_and_label()

            self.image, self.gt = self._fp32([self.image, self.gt])
            self._random_crop()
            self._flip()
            self._rotate()

            if self.random_quality_prob > 0.0:
                self._random_quality()
            if self.rgb_permutation_prob > 0.0:
                self._rgb_permutation()
            if self.brightness_prob > 0.0:
                self._random_brightness()
            if self.contrast_prob > 0.0:
                self._random_contrast()
            if self.hue_prob > 0.0:
                self._random_hue()
            if self.saturation_prob > 0.0:
                self._random_saturation()
            if self.gaussian_noise_prob > 0.0:
                self._random_gaussian_noise()
            if self.shred_prob > 0.0:
                self._random_shred()
            if self.shade_prob > 0.0:
                self._random_shade()
            if self.warp_prob > 0.0:  # todo: Embed the logical part in _warp function
                self.image, self.gt = tf.py_func(self._warp,
                                                 [self.image, self.gt, self.warp_prob, self.warp_ratio, self.warp_crop_prob],
                                                 [tf.float32, tf.float32])
                self.image.set_shape([self.crop_size[0], self.crop_size[1], 3])
                self.gt.set_shape([self.crop_size[0], self.crop_size[1]])
                self.gt = tf.expand_dims(self.gt, 2)
            if self.elastic_distortion_prob > 0.0:
                self.image = tf.py_func(self.draw_grid, [self.image, 5], tf.float32)  # uncomment to visualize
                img_gt_pair = tf.concat([self.image, self.gt], 2)
                img_gt_pair = tf.py_func(self.elastic_transform,
                                         [img_gt_pair, self.elastic_distortion_prob],
                                         tf.float32)

                img_gt_pair.set_shape([self.crop_size[0], self.crop_size[1], 4])
                self.image = img_gt_pair[:, :, :3]
                self.gt = img_gt_pair[:, :, 3]
                self.gt.set_shape([self.crop_size[0], self.crop_size[1]])
                self.gt = tf.expand_dims(self.gt, 2)

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
        b, h, w, c = get_shape(input_tensor)
        mean = [0.485, 0.456, 0.406]
        mean = np.expand_dims(np.expand_dims(mean, 0), 0)
        mean = tf.constant(np.stack([mean] * b, 0), tf.float32)
        std = [0.229, 0.224, 0.225]
        std = np.expand_dims(np.expand_dims(std, 0), 0)
        std = tf.constant(np.stack([std] * b, 0), tf.float32)
        normalized = (input_tensor / 255.0 - mean) / std
        return normalized
