# from functions.project_fn.preprocess_utils import *
from functions.project_fn.preprocess_developing import *


def get_tfrecord_features(task):
    if task == "segmentation":
        return {
            "image/encoded": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "image/filename": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "image/height": tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            "image/width": tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            "image/segmentation/class/encoded": tf.FixedLenFeature(
                (), tf.string, default_value="")
        }
    elif task == "classification":
        return {
            "image/encoded": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "image/filename": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "image/height": tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            "image/width": tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            "image/label": tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            "image/class": tf.FixedLenFeature(
                (), tf.string, default_value=""),
        }
    elif task == "deblur":
        return {
            "image/encoded": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "image/filename": tf.FixedLenFeature(
                (), tf.string, default_value=""),
            "image/height": tf.FixedLenFeature(
                (), tf.int64, default_value=0),
            "image/width": tf.FixedLenFeature(
                (), tf.int64, default_value=0),
        }


def get_data_from_tfrecord(config, drop_remainder=True):
    if not config.batch_size % len(config.gpu_ids) == 0:
        raise ValueError("Batch size must be eqully divided by the number of gpus")

    features = get_tfrecord_features(config.task)
    dataset_path = config.dataset_dir
    dataset_split = config.dataset_split

    def _process_images(image, label):
        processed_image = tf.cast(image, tf.float32)
        if config.task == "deblur":
            if config.blur_kernel_max_size % 2 == 0:
                raise ValueError("blur_kernel_max_size must be odd number")
            if config.blur_kernel_min_size % 2 == 0:
                raise ValueError("blur_kernel_min_size must be odd number")
            if config.blur_kernel_min_size < 3:
                raise ValueError("blur_kernel_min_size must be larger than 3")
            extended_margin = config.blur_kernel_max_size
            if config.do_grayscaling:
                if config.grayscaling_scheme == "grayscaling":
                    processed_image = tf.image.rgb_to_grayscale(processed_image)
                elif config.grayscaling_scheme == "random_channelling":
                    random_channel_idx = tf.random_uniform((), maxval=3, dtype=tf.int32)
                    processed_image = processed_image[:, :, random_channel_idx]
                    processed_image = processed_image[:, :, tf.newaxis]
                else:
                    raise ValueError("unsupported ")

            elif config.input_image_type == "color":
                pass
            else:
                raise ValueError("unsupported input_image_type")
        else:
            extended_margin = 0

        if config.crop_size[0] % 2 == 0:
            raise ValueError("crop height must be odd number")
        if config.crop_size[1] % 2 == 0:
            raise ValueError("crop width must be odd number")

        crop_height = config.crop_size[0] + extended_margin
        crop_width = config.crop_size[1] + extended_margin
        min_scale_factor = config.min_random_scale_factor
        max_scale_factor = config.max_random_scale_factor
        scale_factor_step_size = config.scale_factor_step_size

        if label is not None:
            label = tf.cast(label, tf.int32)

        # scale up or down raw images according to "min_length_limit" and "min_length_limit"
        if config.min_length_limit:
            min_length_limit = float(max(min(crop_height, crop_width), config.min_length_limit))
            processed_image = scale_up_if_too_small(processed_image, min_length_limit)
        if config.max_length_limit:
            max_length_limit = float(max(max(crop_height, crop_width), config.max_length_limit))
            processed_image = scale_down_if_too_large(processed_image, max_length_limit)

        # Data augmentation by randomly scaling the inputs.
        scale = get_random_scale(
            min_scale_factor, max_scale_factor, scale_factor_step_size)
        processed_image, label = randomly_scale_image_and_label(
            processed_image, label, scale)

        # Randomly crop the image and label.
        processed_image, label = random_crop(
            [processed_image, label], crop_height, crop_width)
        if config.task == "deblur":
            if config.input_image_type == "gray":
                processed_image.set_shape([crop_height, crop_width, 1])
        else:
            processed_image.set_shape([crop_height, crop_width, 3])
        if label is not None:
            label.set_shape([crop_height, crop_width, 1])

        # Randomly left-right flip the image and label.
        if label is not None:
            label = flip_or_rotate(processed_image, config)
        processed_image = flip_or_rotate(processed_image, config)
        processed_image = additional_augmentation(processed_image, config)
        return processed_image, label

    def _parse_fn_for_tfrecord(tfrecord):
        # for the key of dictionary,
        #   "X" is input images, which is always the input of network
        #   "Y" is the corresponding ground truth, where
        #       "filename" is the file names of images in "X"
        #       "seg" is the segmentation corresponding to "X" in segmentation task
        #       "label" is the label number corresponding to "X" in classification task
        #       "clsname" is the represented img_name of class corresponding to "label" in classification
        if config.task == "segmentation":
            parsed = tf.parse_single_example(tfrecord, features)
            img = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image/encoded"], channels=3))
            name = tf.convert_to_tensor(parsed["image/filename"])
            seg = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image/segmentation/class/encoded"], channels=1))
            img, seg = _process_images(img, seg)
            return {"X": img, "filename": name}, {"Y": seg}

        elif config.task == "classification":
            parsed = tf.parse_single_example(tfrecord, features)
            label = tf.convert_to_tensor(parsed["image/label"])
            img = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image/encoded"], channels=3))
            name = tf.convert_to_tensor(parsed["image/filename"])
            clsname = tf.convert_to_tensor(parsed["image/class"])
            img, _ = _process_images(img, None)
            return {"X": img, "filename": name}, {"Y": label, "clsname": clsname}

        elif config.task == "deblur":
            parsed = tf.parse_single_example(tfrecord, features)
            img = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image/encoded"], channels=3))
            name = tf.convert_to_tensor(parsed["image/filename"])
            img, _ = _process_images(img, None)
            return {"Y": img, "filename": name}

    def _get_data_files(data_sources):
        if isinstance(data_sources, (list, tuple)):
            data_files = []
            for source in data_sources:
                data_files += _get_data_files(source)
        else:
            if "*" in data_sources or "?" in data_sources or "[" in data_sources:
                data_files = tf.gfile.Glob(data_sources)
            else:
                data_files = [data_sources]
        if not data_files:
            raise ValueError("No data files found in %s" % (data_sources,))
        return data_files

    data_fullpath = dataset_path + "/" + dataset_split + "-*"

    tfrecords = _get_data_files(data_fullpath)
    dataset = tf.data.TFRecordDataset(tfrecords)

    if config.phase == "train":
        dataset = dataset.shuffle(3000)

    dataset = dataset.repeat()
    dataset = dataset.apply(tf.data.experimental.map_and_batch(_parse_fn_for_tfrecord, config.batch_size,
                                                               drop_remainder=drop_remainder))
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
