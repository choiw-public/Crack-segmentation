from functions.project_fn.utils import get_shape as get_shape
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2 as cv
import tensorflow as tf
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

    def _get_random_scale(self):
        if self.random_scale_range[0] < 0:
            raise ValueError("min_scale_factor cannot be nagative value")
        if self.random_scale_range[0] > self.random_scale_range[1]:
            raise ValueError("min_scale_factor must be larger than max_scale_factor")
        elif self.random_scale_range[0] == self.random_scale_range[0]:
            return tf.cast(self.random_scale_range[0], tf.float32)
        else:
            return tf.random_uniform([], minval=self.random_scale_range[0], maxval=self.random_scale_range[0])

    def _randomly_scale_image_and_label(self, image, gt):
        """Randomly scales image and label.

        Args:
          image: Image with original_shape [height, width, 3].
          label: Label with original_shape [height, width, 1].
          scale: The value to scale image and label.

        Returns:
          Scaled image and label.
        """
        scale = self._get_random_scale()
        h, w, c = get_shape(image)
        new_dim = tf.cast(tf.cast([h, w], tf.float32) * scale, tf.int32)

        # Need squeeze and expand_dims because image interpolation takes
        # 4D tensors as input.
        image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), new_dim, align_corners=True), [0])
        gt = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(gt, 0), new_dim, align_corners=True), [0])
        return image, gt

    def _random_crop(self, image, gt):
        # concat in channel
        image_gt_pair = tf.concat([image, gt], 2)
        image_gt_pair_cropped = tf.image.random_crop(image_gt_pair, [self.crop_size[0], self.crop_size[1], 4])
        image = image_gt_pair_cropped[:, :, :3]
        gt = image_gt_pair_cropped[:, :, 3:]
        return image, gt

    def _flip(self, image, gt):
        do_flip = tf.less_equal(tf.random_uniform([]), self.flip_probability)
        image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        gt = tf.cond(do_flip, lambda: tf.image.flip_left_right(gt), lambda: gt)
        return image, gt

    def _rotate(self, image, gt):
        on_off = tf.less_equal(tf.random_uniform([]), self.rotate_probability)
        if self.rotate_angle_by90:
            rotate_k = tf.random_uniform((), maxval=3, dtype=tf.int32)
            image = tf.cond(on_off, lambda: tf.image.rot90(image, rotate_k), lambda: image)
            gt = tf.cond(on_off, lambda: tf.image.rot90(gt, rotate_k), lambda: gt)
        else:
            angle = tf.random.uniform((), minval=self.rotate_angle_range[0], maxval=self.rotate_angle_range[1], dtype=tf.float32)
            image = tf.cond(on_off, lambda: tf.contrib.image.rotate(image, angle, interpolation="BILINEAR"))
            gt = tf.cond(on_off, lambda: tf.contrib.image.rotate(gt, angle, interpolation="NEAREST"))
        return image, gt

    def _random_quality(self, image):
        do_quality = tf.less_equal(tf.random_uniform([]), self.random_quality_prob)
        image = tf.cond(do_quality,
                        lambda: tf.image.random_jpeg_quality(image, self.random_quality[0], self.random_quality[1]),
                        lambda: image)
        image.set_shape([self.crop_size[0], self.crop_size[1], 3])
        return image

    def _rgb_permutation(self, image):
        def execute_fn(image):
            image = tf.transpose(image, [2, 0, 1])
            image = tf.random.shuffle(image)
            return tf.transpose(image, [1, 2, 0])

        do_permutation = tf.less_equal(tf.random_uniform([]), self.rgb_permutation_prob)
        return tf.cond(do_permutation, lambda: execute_fn(image), lambda: image)

    def _random_brightness(self, image):
        do_brightness = tf.less_equal(tf.random_uniform([]), self.brightness_prob)
        delta = tf.random_uniform([], maxval=self.brightness_constant)
        return tf.cond(do_brightness,
                       lambda: tf.image.adjust_brightness(image, delta),
                       lambda: image)

    def _random_contrast(self, image):
        do_contrast = tf.less_equal(tf.random_uniform([]), self.contrast_prob)
        contrast_factor = tf.random_uniform([], minval=self.contrast_constant[0], maxval=self.contrast_constant[1])
        return tf.cond(do_contrast,
                       lambda: tf.image.adjust_contrast(image, contrast_factor),
                       lambda: image)

    def _random_hue(self, image):
        do_hue = tf.less_equal(tf.random_uniform([]), self.hue_prob)
        delta = tf.random_uniform([], minval=self.hue_constant[0], maxval=self.hue_constant[1])
        return tf.cond(do_hue,
                       lambda: tf.image.adjust_hue(image, delta),
                       lambda: image)

    def _random_saturation(self, image):
        do_saturation = tf.less_equal(tf.random_uniform([]), self.saturation_prob)
        saturation_factor = tf.random_uniform([], minval=self.saturation_constant[0], maxval=self.saturation_constant[1])
        return tf.cond(do_saturation,
                       lambda: tf.image.adjust_saturation(image, saturation_factor),
                       lambda: image)

    def _random_gaussian_noise(self, image):
        def execute_fn(image, std):
            image = image / 255.0
            rnd_stddev = tf.random_uniform([], minval=std[0], maxval=std[1])
            noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=rnd_stddev, dtype=tf.float32)
            return tf.clip_by_value(image + noise, 0.0, 1.0) * 255.0

        do_gaussian_noise = tf.less_equal(tf.random_uniform([]), self.gaussian_noise_prob)
        return tf.cond(do_gaussian_noise,
                       lambda: execute_fn(image, self.gaussian_noise_std),
                       lambda: image)

    def _random_shred(self, image, gt):
        # todo: make this function work for tensors
        def execute_fn(image, gt, shred_range):
            image_shape = get_shape(image)
            gt_shape = get_shape(gt)
            shred_num = tf.random.uniform([], minval=shred_range[0], maxval=shred_range[1] + 1, dtype=tf.uint8)
            for split_axis in [0, 1]:
                split_indices = np.linspace(0, image_shape[split_axis], shred_num + 1, dtype=np.int32)
                # tf.linspace(0, image_shape[split_axis], shred_num+1)
                split_indices = split_indices[1:] - split_indices[:-1]
                splitted_image = tf.split(image, split_indices, split_axis)
                splitted_gt = tf.split(gt, split_indices, split_axis)
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
        return tf.cond(do_shred,
                       lambda: execute_fn(image, gt, self.shred_piece_range),
                       lambda: image)

    def _random_shade(self, image):
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
        return tf.cond(do_shade, lambda: execute_fn(shade_src, image), lambda: image)

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

    def preprocessing(self, image, gt):
        if image is None:
            raise ValueError("image should not be none")
        if gt is None:
            raise ValueError("gt should not be none in training")

            # Data augmentation by randomly scaling the inputs.
        if self.random_scale_range != [1.0, 1.0] and self.random_scale_range is not None:
            image, gt = self._randomly_scale_image_and_label(image, gt)

        image, gt = self._fp32([image, gt])
        image, gt = self._random_crop(image, gt)

        if self.flip_probability > 0:
            image, gt = self._flip(image, gt)
        if self.rotate_probability > 0:
            image, gt = self._rotate(image, gt)
        if self.random_quality_prob > 0.0:
            image = self._random_quality(image)
        if self.rgb_permutation_prob > 0.0:
            image = self._rgb_permutation(image)
        if self.brightness_prob > 0.0:
            image = self._random_brightness(image)
        if self.contrast_prob > 0.0:
            image = self._random_contrast(image)
        if self.hue_prob > 0.0:
            image = self._random_hue(image)
        if self.saturation_prob > 0.0:
            image = self._random_saturation(image)
        if self.gaussian_noise_prob > 0.0:
            image = self._random_gaussian_noise(image)
        if self.shred_prob > 0.0:
            image, gt = self._random_shred(image, gt)
        if self.shade_prob > 0.0:
            image = self._random_shade(image)
        if self.warp_prob > 0.0:
            image, gt = tf.py_func(self._warp,
                                   [image, gt, self.warp_prob, self.warp_ratio, self.warp_crop_prob],
                                   [tf.float32, tf.float32])
            image.set_shape([self.crop_size[0], self.crop_size[1], 3])
            gt.set_shape([self.crop_size[0], self.crop_size[1]])
            gt = tf.expand_dims(gt, 2)
            # todo: unexpected gt tensor shape. should be fixed
        if self.elastic_distortion_prob > 0.0:
            image = tf.py_func(self.draw_grid, [image, 5], tf.float32)  # uncomment to visualize
            img_gt_pair = tf.concat([image, gt], 2)
            img_gt_pair = tf.py_func(self.elastic_transform,
                                     [img_gt_pair, self.elastic_distortion_prob],
                                     tf.float32)

            img_gt_pair.set_shape([self.crop_size[0], self.crop_size[1], 4])
            image = img_gt_pair[:, :, :3]
            gt = img_gt_pair[:, :, 3]
            gt.set_shape([self.crop_size[0], self.crop_size[1]])
            gt = tf.expand_dims(gt, 2)
        return image, gt
