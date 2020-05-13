from functions.project_fn.preprocess_developing import Preprocessing
from functions.project_fn.misc_utils import list_getter
import tensorflow as tf
from natsort import natsorted
import multiprocessing
import os


class InputPipeline(Preprocessing):
    def __init__(self, config):
        self.tfrecord_feature = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                                 "filename": tf.FixedLenFeature((), tf.string, default_value=""),
                                 "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                                 "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                                 "segmentation": tf.FixedLenFeature((), tf.string, default_value="")}
        super(InputPipeline, self).__init__(config)
        self.drop_remainder = True if self.phase == "Train" else False
        self.config = config

    def __getattr__(self, item):
        try:
            return getattr(self.config, item)
        except AttributeError:
            raise AttributeError("'config' has no attribute '%s'" % item)

    def _tfrecord_parser(self, tfrecord):
        parsed = tf.parse_single_example(tfrecord, self.tfrecord_feature)
        img = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=3))
        fname = tf.convert_to_tensor(parsed["filename"])
        seg = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["segmentation"], channels=1))
        img, seg = self.preprocessing(img, seg)
        return {"input": img, "filename": fname, "gt": seg}

    def _get_batch_and_init(self, tfrecord_dir, batch_size):
        tfrecord_list = list_getter(tfrecord_dir, extension="tfrecord")
        data = tf.data.TFRecordDataset(tfrecord_list)
        if self.is_train:
            data = data.repeat()
        data = data.shuffle(batch_size * 10)
        data = data.map(self._tfrecord_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size, drop_remainder=self.drop_remainder)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        iterator = data.make_one_shot_iterator()
        return iterator.get_next(), iterator.initializer

    def _input_from_tfrecord(self):
        if self.background_dir:
            if not 1.0 >= self.background_proportion > 0.0:
                raise ValueError("Unexpected background_proportion: %s" % self.background_proportion)
            batch_size_background = int(self.batch_size * self.background_proportion)
        else:
            batch_size_background = 0

        if self.blur_dir:
            if not 1.0 >= self.blur_proportion > 0.0:
                raise ValueError("Unexpected blur_proportion: %s" % self.blur_proportion)
            batch_size_blur = int(self.batch_size * self.blur_proportion)
        else:
            batch_size_blur = 0
        batch_size_main = self.batch_size - batch_size_background - batch_size_blur
        if batch_size_main < 0:
            raise ValueError("Unexpected batch_size_main: %s" % batch_size_main)

        # def parse_fn(tfrecord):
        #     parsed = tf.parse_single_example(tfrecord, self.tfrecord_feature)
        #     img = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=3))
        #     fname = tf.convert_to_tensor(parsed["filename"])
        #     seg = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["segmentation"], channels=1))
        #     img, seg = self._preprocessing(img, seg)
        #     return {"input": img, "filename": fname, "gt": seg}

        # for main images
        batch_main, init_main = self._get_batch_and_init(self.dataset_dir, batch_size_main)

        # for background images
        if batch_size_background > 0:
            batch_background, init_background = self._get_batch_and_init(self.background_dir, batch_size_background)
        else:
            batch_background = None
            init_background = None
        if batch_size_blur > 0:
            batch_blur, init_blur = self._get_batch_and_init(self.blur_dir, batch_size_blur)
        else:
            batch_blur = None
            init_blur = None

        keys = batch_main.keys()
        batch_whole = dict()
        for key in keys:
            value = [batch_main[key]]
            if batch_background:
                value.append(batch_background[key])
            if batch_blur:
                value.append(batch_blur[key])
            batch_whole[key] = tf.concat(value, 0)

        indices = tf.range(start=0, limit=self.batch_size, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        for key in keys:
            batch_whole[key] = tf.gather(batch_whole[key], shuffled_indices)
        whole_init = tf.group([init_main, init_blur, init_background])
        return batch_whole, whole_init

    def _tfrecord_pipeline(self):
        print("=============================== Attention ===============================")
        print("Building input pipeline with tfrecord...")
        in_data, init = self._input_from_tfrecord()
        if tf.executing_eagerly():
            return in_data
        else:
            return in_data, init

    def build(self):
        if self.phase == "train":
            return self._tfrecord_pipeline()
        elif self.phase in ["eval", "vis"]:
            if self.data_type == "image":
                return self._image_input_pipeline()
            elif self.data_type == "tfrecord":
                return self._tfrecord_input_pipeline()
            else:
                raise ValueError("not supported")

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
