import collections
import os.path
import tensorflow as tf

slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder

_ITEMS_TO_DESCRIPTIONS = {
    "image": "A color image of varying height and width.",
    "labels_class": ("A semantic segmentation label whose size matches image."
                     "Its values range from 0 (background) to num_classes."),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    "DatasetDescriptor",
    ["splits_to_sizes",  # Splits of the dataset into training, val, and test.
     "num_classes",  # Number of semantic classes.
     "ignore_label",  # Ignore label value.
     ]
)

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        "_train": 2975,
        "val": 500,
    },
    num_classes=19,
    ignore_label=255,
)

_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        "_train": 1464,
        "trainval": 2913,
        "val": 1449,
    },
    num_classes=21,
    ignore_label=255,
)


_CITYSCAPE_ONLY_POLE_AS_OBJECT = DatasetDescriptor(
    splits_to_sizes={
        "_train": 2975,
        "trainval": 3475,
        "val": 500,
    },
    num_classes=2,
    ignore_label=255,
)

_CIVIL_ONLY_FUSED_CRACK = DatasetDescriptor(
    splits_to_sizes={
        "_train": 3079,
        "trainval": 3837,
        "val": 768,
    },
    num_classes=2,
    ignore_label=255,
)

_CIVIL_CRACK_SIMPLE = DatasetDescriptor(
    splits_to_sizes={
        "_train": 257,
        "trainval": 58,
        "val": 315,
    },
    num_classes=2,
    ignore_label=255,
)


_DATASETS_INFORMATION = {
    "cityscapes": _CITYSCAPES_INFORMATION,
    "pascal_voc_seg": _PASCAL_VOC_SEG_INFORMATION,
    "civil_only_fused_crack": _CIVIL_ONLY_FUSED_CRACK,
    "civil_crack_simple": _CIVIL_CRACK_SIMPLE,
    "cityscape_only_pole_as_object": _CITYSCAPE_ONLY_POLE_AS_OBJECT
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = "%s-*"


def get_dataset(dataset_name, split_name, dataset_dir):
    """Gets an instance of slim Dataset.

    Args:
      dataset_name: Dataset img_name.
      split_name: A _train/val Split img_name.
      dataset_dir: The directory of the dataset sources.

    Returns:
      An instance of slim Dataset.

    Raises:
      ValueError: if the dataset_name or split_name is not recognized.
    """
    if dataset_name not in _DATASETS_INFORMATION:
        raise ValueError("The specified dataset is not supported yet.")

    splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

    if split_name not in splits_to_sizes:
        raise ValueError("data_pipeline split img_name %s not recognized" % split_name)

    # Prepare the variables for different datasets.
    num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
    ignore_label = _DATASETS_INFORMATION[dataset_name].ignore_label

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
        name=dataset_name,
        multi_label=True)
