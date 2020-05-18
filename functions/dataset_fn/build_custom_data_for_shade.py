import re
import math
import os.path
import build_data
import tensorflow as tf
import time
from random import shuffle

FLAGS = tf.app.flags.FLAGS
shade_folder_dir = "../../shades/source"
tfrecord_out_dir = "../../shades/tfrecord"

dataset_split_name = "train"
num_shards = 1


def get_img_list(image_folder):
    img_list = []
    for path, subdirs, files in os.walk(image_folder):
        for name in files:
            if name.lower().endswith(("png", "jpg")):
                img_list.append(os.path.join(path, name))
    img_list.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])
    return img_list


def decoder_getter(extension):
    if extension == "jpg":
        return tf.image.decode_jpeg
    elif extension == "png":
        return tf.image.decode_png
    else:
        raise ValueError("unknown extionsion: %s" % extension)


def get_unique_file_extension(file_list):
    file_extension = []
    for entry in file_list:
        basename = os.path.basename(entry)
        file_extension.append(os.path.basename(basename).split(".")[-1])
    file_extension = list(set(file_extension))
    if len(file_extension) == 1:
        return file_extension[0]
    else:
        raise ValueError("standardization of image file type is required")


img_list = get_img_list(shade_folder_dir)
if not img_list:
    raise ValueError("no images exist in the specified folder")

img_filetype = get_unique_file_extension(img_list)

num_images = len(img_list)
num_per_shard = int(math.ceil(num_images / float(num_shards)))

sharp_img_reader = build_data.ImageReader(img_filetype, channels=3)
if not os.path.exists(tfrecord_out_dir):
    os.makedirs(tfrecord_out_dir)

for shard_id in range(num_shards):
    output_filename = os.path.join(tfrecord_out_dir, "%s-%05d-of-%05d.tfrecord" % (dataset_split_name, shard_id + 1, num_shards))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id + 1) * num_per_shard, num_images)
        rnd_idx = range(0, num_images)
        shuffle(rnd_idx)
        t = time.time()
        for i in range(start_idx, end_idx):
            img_id = img_list[rnd_idx[i]]
            # Read the image.
            img_subfolder = os.path.dirname(img_id)
            img_filename = os.path.join(img_subfolder, os.path.basename(img_id))
            img_data = tf.gfile.GFile(img_filename, "r").read()
            img_height, img_width, img_channel = sharp_img_reader.read_image_dims(img_data)

            # Convert to tf example.
            file_id = os.path.basename(img_filename)
            example = build_data.shade_source_to_tfexample(img_data, file_id, img_height, img_width)
            tfrecord_writer.write(example.SerializeToString())
            if (i + 1) % 100 == 0:
                print("Converting image %d/%d (shard %d) [%.4f seg/100]" % (i + 1, num_images, shard_id + 1, time.time() - t))
                t = time.time()
