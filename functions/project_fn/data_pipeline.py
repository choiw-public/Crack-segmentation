from functions.project_fn.preprocess import Preprocessing
from functions.project_fn.utils import list_getter
import tensorflow as tf
import os


class DataPipeline(Preprocessing):
    def __init__(self, config):
        self.tfrecord_feature = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                                 "filename": tf.FixedLenFeature((), tf.string, default_value=""),
                                 "height": tf.FixedLenFeature((), tf.int64, default_value=0),
                                 "width": tf.FixedLenFeature((), tf.int64, default_value=0),
                                 "segmentation": tf.FixedLenFeature((), tf.string, default_value="")}
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
        image = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=3))
        gt = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["segmentation"], channels=1))
        image, gt = self.preprocessing(image, gt)
        return {"input_data": image, "gt": gt, "filename": fname}

    @staticmethod
    def _image_gt_parser(image_name, gt_name):
        image = tf.image.decode_png(tf.read_file(image_name), 3)
        gt = tf.image.decode_png(tf.read_file(gt_name), 1)
        return {"input_data": image, "gt": gt, "filename": image_name}

    @staticmethod
    def _image_parser(image_name):
        return {"input_data": tf.image.decode_png(tf.read_file(image_name), 3), "filename": image_name}

    def _get_batch_and_init(self, tfrecord_dir, batch_size):
        tfrecord_list = list_getter(tfrecord_dir, extension="tfrecord")
        if not tfrecord_list:
            raise ValueError("tfrecord does not exist: %s" % tfrecord_dir)
        data = tf.data.TFRecordDataset(tfrecord_list)
        data = data.repeat()
        data = data.shuffle(batch_size * 10)
        data = data.map(self._tfrecord_parser, 4).batch(batch_size, self._drop_remainder)
        data = data.prefetch(4)  # tf.data_pipeline.experimental.AUTOTUNE
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

        if batch_background or batch_blur:
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
            self.input_data = batch_whole["image"]
            self.gt = batch_whole["gt"]
        else:
            self.input_data = batch_main["image"]
            self.gt = batch_main["gt"]
        self.data_init = None

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

        img_list = list_getter(self.img_dir, "jpg")
        img_list_tensor = tf.convert_to_tensor(img_list, dtype=tf.string)
        img_data = tf.data.Dataset.from_tensor_slices(img_list_tensor)
        if self.phase == "eval":
            gt_list = list_getter(self.seg_dir, "png")
            inspect_pairness(gt_list, img_list)
            inspect_file_extension(gt_list)
            inspect_file_extension(img_list)
            gt_list_tensor = tf.convert_to_tensor(gt_list, dtype=tf.string)
            gt_data = tf.data.Dataset.from_tensor_slices(gt_list_tensor)
            data = tf.data.Dataset.zip((img_data, gt_data))
            data = data.map(self._image_gt_parser, 4).batch(self.batch_size, False)
        else:
            data = img_data.map(self._image_parser, 4).batch(self.batch_size, False)
        data = data.prefetch(4)  # tf.data_pipeline.experimental.AUTOTUNE
        iterator = data.make_initializable_iterator()
        dataset = iterator.get_next()
        self.input_data = dataset["input_data"]
        self.gt = dataset["gt"] if self.phase == "eval" else None
        self.filename = dataset["filename"]
        self.data_init = iterator.initializer

    def _build_input_pipeline(self):
        if self.phase == "train":
            self._input_from_tfrecord()
        elif self.phase == "eval":
            self._input_from_image()
        elif self.phase == "vis":
            if self.data_type == "image":
                self._input_from_image()
            elif self.data_type == "video":
                # input_data and gt will be handled by ModelHandler
                self.input_data = tf.placeholder(tf.float32, [1, None, None, 3])
                self.gt = None
                self.filename = None
                self.data_init = None
