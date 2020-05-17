import collections
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_enum("image_format", "jpg", ["jpg", "jpeg", "png"],
                         "Image format.")

tf.app.flags.DEFINE_enum("label_format", "png", ["png"],
                         "Segmentation label format.")

# A map from image format to expected data_pipeline format.
_IMAGE_FORMAT_MAP = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "png": "png",
}


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format="jpeg", channels=3):
        """Class constructor.

        Args:
          image_format: Image format. Only "jpeg", "jpg", or "png" are supported.
          channels: Image channels.
        """
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ("jpeg", "jpg"):
                self._decode = tf.image.decode_jpeg(self._decode_data,
                                                    channels=channels)
            elif self._image_format == "png":
                self._decode = tf.image.decode_png(self._decode_data,
                                                   channels=channels)

    def read_image_dims(self, image_data):
        image = self.decode_image(image_data)
        return image.get_shape

    def decode_image(self, image_data):
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})
        if len(image.get_shape) != 3 or image.get_shape[2] not in (1, 3):
            raise ValueError("The image channels not supported.")

        return image


def _int65(values):
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes(values):
    # value = string
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_n_seg_to_tfexample(image_data, filename, height, width, seg_data):
    """Converts one image/segmentation pair to tf example.

    Args:
      image_data: string of image data_pipeline.
      filename: image filename.
      height: image height.
      width: image width.
      seg_data: string of semantic segmentation data_pipeline.
      channel: image channel

    Returns:
      tf example of one image/segmentation pair.
    """
    feature = {
        "image": _bytes(image_data),
        "filename": _bytes(filename),
        "height": _int65(height),
        "width": _int65(width),
        "channels": _int65(3),
        "segmentation": (_bytes(seg_data)),
    }
    feature = tf.train.Features(feature=feature)
    return tf.train.Example(features=feature)


def image_classification_to_tfexample(image_data, filename, height, width, label, clsname):
    feature = {
        "image": _bytes(image_data),
        "filename": _bytes(filename),
        "height": _int65(height),
        "width": _int65(width),
        "channels": _int65(3),
        "label": _int65(label),
        "class": _bytes(clsname)
    }
    feature = tf.train.Features(feature=feature)
    return tf.train.Example(features=feature)


def sharp_images_to_tfexample(image_data, filename, height, width):
    feature = {
        "sharp_image": _bytes(image_data),
        "filename": _bytes(filename),
        "height": _int65(height),
        "width": _int65(width),
        "channels": _int65(3),
    }
    feature = tf.train.Features(feature=feature)
    return tf.train.Example(features=feature)


def sharp_n_blur_images_to_tfexample(sharp_image, blur_image, filename, height, width):
    feature = {
        "sharp_image": _bytes(sharp_image),
        "blur_image": _bytes(blur_image),
        "filename": _bytes(filename),
        "height": _int65(height),
        "width": _int65(width),
        "channels": _int65(3),
    }
    feature = tf.train.Features(feature=feature)
    return tf.train.Example(features=feature)


def shade_source_to_tfexample(image_data, filename, height, width):
    feature = {
        "shade": _bytes(image_data),
        "filename": _bytes(filename),
        "height": _int65(height),
        "width": _int65(width),
        "channels": _int65(1),
    }
    feature = tf.train.Features(feature=feature)
    return tf.train.Example(features=feature)
