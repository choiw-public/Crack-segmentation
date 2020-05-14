from functions.project_fn.preprocess_developing import Preprocessing
from functions.project_fn.misc_utils import list_getter
import tensorflow as tf
import os


class InputPipeline(Preprocessing):
    def __init__(self, config):
        self.tfrecord_feature = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                                 "filename": tf.FixedLenFeature((), tf.string, default_value=""),
                                 "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                                 "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                                 "segmentation": tf.FixedLenFeature((), tf.string, default_value="")}
        super(InputPipeline, self).__init__(config)
        self.config = config
        self._drop_remainder = True if self.phase == "Train" else False
        self._build_input_pipeline()

    def __getattr__(self, item):
        try:
            return getattr(self.config, item)
        except AttributeError:
            raise AttributeError("'config' has no attribute '%s'" % item)

    def _tfrecord_parser(self, data):
        parsed = tf.parse_single_example(data, self.tfrecord_feature)
        fname = tf.convert_to_tensor(parsed["filename"])
        self.image = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=3))
        self.gt = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["segmentation"], channels=1))
        self.preprocessing()
        return {"input": self.image, "filename": fname, "gt": self.gt}

    def _image_parser(self, image_name, gt_name):
        self.image = tf.image.decode_png(tf.read_file(image_name), 3)
        self.gt = tf.image.decode_png(tf.read_file(gt_name), 1)
        self.preprocessing()
        return {"input": self.image, "gt": self.gt}

    def _get_batch_and_init(self, tfrecord_dir, batch_size):
        tfrecord_list = list_getter(tfrecord_dir, extension="tfrecord")
        if not tfrecord_list:
            raise ValueError("tfrecord does not exist: %s" % tfrecord_dir)
        data = tf.data.TFRecordDataset(tfrecord_list)
        if self.is_train:
            data = data.repeat()
        data = data.shuffle(batch_size * 10)
        data = data.map(self._tfrecord_parser, tf.data.experimental.AUTOTUNE).batch(batch_size, self._drop_remainder)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        iterator = data.make_one_shot_iterator()
        return iterator.get_next()

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

        # for main images
        batch_main = self._get_batch_and_init(self.dataset_dir, batch_size_main)

        # for background images
        if batch_size_background > 0:
            batch_background = self._get_batch_and_init(self.background_dir, batch_size_background)
        else:
            batch_background = None
        if batch_size_blur > 0:
            batch_blur = self._get_batch_and_init(self.blur_dir, batch_size_blur)
        else:
            batch_blur = None

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
        self.data = batch_whole

    def _build_input_pipeline(self):
        if self.phase == "train":
            self._input_from_tfrecord()
        elif self.phase in ["eval", "vis"]:
            if self.data_type == "image":
                self._input_from_image()
            elif self.data_type == "tfrecord":
                return self._input_from_tfrecord()
            else:
                raise ValueError("not supported")

    def _input_from_image(self):
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

        img_list = list_getter(self.img_dir, ("png", "jpg"))
        gt_list = list_getter(self.seg_dir, "png")
        inspect_pairness(gt_list, img_list)
        inspect_file_extension(gt_list)
        inspect_file_extension(img_list)

        img_list = tf.convert_to_tensor(img_list, dtype=tf.string)
        seg_list = tf.convert_to_tensor(gt_list, dtype=tf.string)
        img_data = tf.data.Dataset.from_tensor_slices(img_list)
        seg_data = tf.data.Dataset.from_tensor_slices(seg_list)
        data = tf.data.Dataset.zip((img_data, seg_data))
        data = data.map(self._image_parser, tf.data.experimental.AUTOTUNE).batch(self.batch_size, False)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        iterator = data.make_initializable_iterator()
        self.data = iterator.get_next()
        self.init = iterator.initializer
