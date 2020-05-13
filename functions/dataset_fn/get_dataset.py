import os.path
import tensorflow as tf
import imp
slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = "%s-*"

_ITEMS_TO_DESCRIPTIONS = {
    "image": "A color image of varying height and width.",
    "labels_class": ("A semantic segmentation label whose size matches image."
                     "Its values range from 0 (background) to num_classes."),
}



def get_dataset(split_name, dataset_dir):
    dataset_info_path = os.path.join(dataset_dir, "info")
    dataset_info = imp.load_source("info", os.path.join(dataset_info_path, "dataset_info.py")).info
    splits_to_sizes = dataset_info.splits_to_sizes
    if split_name not in splits_to_sizes:
        raise ValueError("data split img_name %s not recognized" % split_name)

    # Prepare the variables for different datasets.
    num_classes = dataset_info.num_classes
    ignore_label = dataset_info.ignore_label

    file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Specify how the TF-Examples are decoded.
    keys_to_features = {
        "image/encoded": tf.FixedLenFeature(
            (), tf.string, default_value=""),
        "image/filename": tf.FixedLenFeature(
            (), tf.string, default_value=""),
        "image/format": tf.FixedLenFeature(
            (), tf.string, default_value="jpeg"),
        "image/height": tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        "image/width": tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        "image/segmentation/class/encoded": tf.FixedLenFeature(
            (), tf.string, default_value=""),
        "image/segmentation/class/format": tf.FixedLenFeature(
            (), tf.string, default_value="png"),
    }
    items_to_handlers = {
        "image": tfexample_decoder.Image(
            image_key="image/encoded",
            format_key="image/format",
            channels=3),
        "image_name": tfexample_decoder.Tensor("image/filename"),
        "height": tfexample_decoder.Tensor("image/height"),
        "width": tfexample_decoder.Tensor("image/width"),
        "labels_class": tfexample_decoder.Image(
            image_key="image/segmentation/class/encoded",
            format_key="image/segmentation/class/format",
            channels=1),
    }

    decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=splits_to_sizes[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        ignore_label=ignore_label,
        num_classes=num_classes,
        name=dataset_info.dataset_name,
        multi_label=True)
