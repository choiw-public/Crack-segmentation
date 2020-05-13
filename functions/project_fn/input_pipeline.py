from functions.project_fn.preprocess_developing import *
from functions.project_fn.misc_utils import list_getter
from natsort import natsorted
import multiprocessing
import os


def get_tfrecord_features():
    return {"image": tf.FixedLenFeature((), tf.string, default_value=""),
            "filename": tf.FixedLenFeature((), tf.string, default_value=""),
            "height": tf.FixedLenFeature((), tf.int64, default_value=0),
            "width": tf.FixedLenFeature((), tf.int64, default_value=0),
            "segmentation": tf.FixedLenFeature((), tf.string, default_value="")}


def input_from_tfrecord(config, drop_remainder=True):
    features = get_tfrecord_features()

    if config.background_dir:
        if config.background_proportion > 0.0:
            batch_background = int(config.batch_size * config.background_proportion)
            batch_actual = config.batch_size - batch_background

        else:
            raise ValueError("unexpected background_proportion")
    else:
        batch_actual = config.batch_size
        batch_background = 0

    if config.blur_dir:
        if config.blur_proportion > 0.0:
            batch_blur = int(config.batch_size * config.blur_proportion)
            batch_actual = batch_actual - batch_blur
        else:
            raise ValueError("unexpected blur_proportion")
    else:
        batch_blur = 0

    def parse_fn(tfrecord):
        parsed = tf.parse_single_example(tfrecord, features)
        img = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=3))
        fname = tf.convert_to_tensor(parsed["filename"])
        seg = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["segmentation"], channels=1))
        img, seg = image_preprocess_for_segmentation(img, seg, config)
        return {"input": img, "filename": fname, "gt": seg}

    # for actual images
    tfrecords_list_actual = list_getter(config.dataset_dir, extension="tfrecord")
    data_actual = tf.data.TFRecordDataset(tfrecords_list_actual)
    if config.is_train:
        data_actual = data_actual.repeat()
    data_actual = data_actual.shuffle(batch_actual * 10)
    data_actual = data_actual.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_actual, drop_remainder=drop_remainder)
    data_actual = data_actual.prefetch(tf.data.experimental.AUTOTUNE)
    if not tf.executing_eagerly():
        iterator_actual = data_actual.make_initializable_iterator()
    else:
        iterator_actual = data_actual.make_one_shot_iterator()
    whole_batch = iterator_actual.get_next()
    whole_init = iterator_actual.initializer
    # for backgorund images
    if batch_background > 0:
        tfrecords_list_background = list_getter(config.background_dir, extension="tfrecord")
        data_background = tf.data.TFRecordDataset(tfrecords_list_background)
        if config.is_train:
            data_background = data_background.repeat()
        data_background = data_background.shuffle(batch_background * 10)
        data_background = data_background.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_background, drop_remainder=drop_remainder)
        data_background = data_background.prefetch(tf.data.experimental.AUTOTUNE)
        if not tf.executing_eagerly():
            iterator_background = data_background.make_initializable_iterator()
        else:
            iterator_background = data_background.make_one_shot_iterator()
        background_batch = iterator_background.get_next()
        background_init = iterator_background.initializer
    else:
        background_batch = None
        background_init = None
    if batch_blur > 0:
        tfrecords_list_blur = list_getter(config.blur_dir, extension="tfrecord")
        data_blur = tf.data.TFRecordDataset(tfrecords_list_blur)
        if config.is_train:
            data_blur = data_blur.repeat()
        data_blur = data_blur.shuffle(batch_background * 10)
        data_blur = data_blur.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_blur, drop_remainder=drop_remainder)
        data_blur = data_blur.prefetch(tf.data.experimental.AUTOTUNE)
        if not tf.executing_eagerly():
            iterator_blur = data_blur.make_initializable_iterator()
        else:
            iterator_blur = data_blur.make_one_shot_iterator()
        blur_batch = iterator_blur.get_next()
        blur_init = iterator_blur.initializer
    else:
        blur_batch = None
        blur_init = None

    keys = whole_batch.keys()
    for key in keys:
        if background_batch:
            whole_batch[key] = tf.concat([whole_batch[key], background_batch[key]], 0)
        if blur_batch:
            whole_batch[key] = tf.concat([whole_batch[key], blur_batch[key]], 0)
    indices = tf.range(start=0, limit=config.batch_size, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    for key in keys:
        whole_batch[key] = tf.gather(whole_batch[key], shuffled_indices)
    if background_init:
        whole_init = tf.group([whole_init, background_init])
    if blur_init:
        whole_init = tf.group([whole_init, blur_init])
    return whole_batch, whole_init


def tfrecord_input_pipeline(config):
    print("=============================== Attention ===============================")
    print("Building input pipeline with tfrecord...")
    in_data, init = input_from_tfrecord(config)
    # if config.blur_aug_probability > 0.0:
    #     raise ValueError("not implemented yet")
    #     with tf.device("/device:GPU:%d" % gpu_id):
    #         return create_blur_images_individual(parsed_data, config)
    if tf.executing_eagerly():
        return in_data
    else:
        return in_data, init
    ########################################################################
    # TODO: for debugging
    # tf.add_to_collection("input", tf.identity(input_for_clones[0]["input"], "input"))
    # tf.add_to_collection("gt", tf.identity(ground_truth_for_clones[0]["gt"], "gt"))
    # tf.add_to_collection("filename", tf.identity(input_for_clones[0]["filename"], "filename"))
    ########################################################################


def get_image_list(config):
    def _list_getter(dir_name):
        image_list = []
        if dir_name:
            for path, subdirs, files in os.walk(dir_name):
                for name in files:
                    if name.lower().endswith(("png", "jpg")):
                        image_list.append(os.path.join(path, name))
            image_list = natsorted(image_list)
        return image_list

    def inspect_file_extension(target_list):
        extensions = list(set([os.path.basename(img_name).split(".")[-1] for img_name in target_list]))
        if len(extensions) > 1:
            raise ValueError("Multiple image formats are used:")
        elif len(extensions) == 0:
            raise ValueError("no image files exist")

    def inspect_pairness(list1, list2):
        if not len(list1) == len(list2):
            raise ValueError("number of images are different")
        for file1, file2 in zip(list1, list2):
            file1_name = os.path.basename(file1).split(".")[-2]
            file2_name = os.path.basename(file2).split(".")[-2]
            if not file1_name == file2_name:
                raise ValueError("image names are different: %s | %s" % (file2, file1))

    seg_list = _list_getter(config.seg_dir)
    img_list = _list_getter(config.img_dir)
    inspect_pairness(seg_list, img_list)
    inspect_file_extension(seg_list)
    inspect_file_extension(img_list)
    return {"image": img_list, "gt": seg_list}


def input_from_image(image_list, config):
    def get_parsed_data(data, parser_fn):
        num_threads = multiprocessing.cpu_count()
        return data.map(parser_fn, num_threads).batch(config.batch_size, drop_remainder=False).make_initializable_iterator()

    def parser(img_id, seg_id):
        img = tf.image.decode_png(tf.read_file(img_id), 3)
        seg = tf.image.decode_png(tf.read_file(seg_id), 1)
        img, seg = image_preprocess_for_segmentation(img, seg, config)
        return {"input": img, "filename": img_id, "gt": seg}

    img_list = tf.convert_to_tensor(image_list["image"], dtype=tf.string)
    seg_list = tf.convert_to_tensor(image_list["gt"], dtype=tf.string)
    img_data = tf.data.Dataset.from_tensor_slices(img_list)
    seg_data = tf.data.Dataset.from_tensor_slices(seg_list)
    parsed_data = get_parsed_data(tf.data.Dataset.zip((img_data, seg_data)), parser)
    return parsed_data.get_next(), parsed_data.initializer


def image_input_pipeline(config):
    # in case of not training phase, images are directly used.
    print("=============================== Attention ===============================")
    print("Building input pipeline with image...")
    image_list = get_image_list(config)
    return input_from_image(image_list, config)


def build_input_pipeline(config):
    if config.phase == "train":
        return tfrecord_input_pipeline(config)
    elif config.phase in ["eval", "vis"]:
        if config.data_type == "image":
            return image_input_pipeline(config)
        elif config.data_type == "tfrecord":
            return tfrecord_input_pipeline(config)
        else:
            raise ValueError("not supported")
