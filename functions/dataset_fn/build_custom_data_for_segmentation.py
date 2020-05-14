import re
import math
import os.path
import tensorflow as tf
import time
from random import shuffle
import build_data

FLAGS = tf.app.flags.FLAGS
img_folder_dir = "../../datasets/toy/img"
seg_folder_dir = "../../datasets/toy/seg"
tfrecord_out_dir = "../../datasets/toy/tfrecord"
min_length = 0

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


img_list = get_img_list(img_folder_dir)
if not img_list:
    raise ValueError("no images exist in the specified folder")
seg_list = get_img_list(seg_folder_dir)
if not seg_list:
    raise ValueError("no images exist in the specified folder")

if not len(img_list) == len(seg_list):
    raise ValueError("different file number of image and label")
for img_name, seg_name in zip(img_list, seg_list):
    img_name = img_name.replace(img_folder_dir, "").split(".")[0]
    seg_name = seg_name.replace(seg_folder_dir, "").split(".")[0]
    if not img_name == seg_name:
        raise ValueError("not paired image and seg names:\n"
                         "image img_name: %s, seg img_name: %s" % (img_name, seg_name))

img_filetype = get_unique_file_extension(img_list)
seg_filetype = get_unique_file_extension(seg_list)

num_images = len(img_list)
num_per_shard = int(math.ceil(num_images / float(num_shards)))

img_reader = build_data.ImageReader(img_filetype, channels=3)
seg_reader = build_data.ImageReader(seg_filetype, channels=1)
if not os.path.exists(tfrecord_out_dir):
    os.makedirs(tfrecord_out_dir)

for shard_id in range(num_shards):
    output_filename = os.path.join(tfrecord_out_dir, "%s-%05d-of-%05d.tfrecord" % (dataset_split_name, shard_id + 1, num_shards))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id + 1) * num_per_shard, num_images)
        rnd_idx = list(range(0, num_images))
        shuffle(rnd_idx)
        t = time.time()
        for i in range(start_idx, end_idx):
            img_id = img_list[rnd_idx[i]]
            # Read the image.
            img_subfolder = os.path.dirname(img_id)
            seg_subfolder = img_subfolder.replace(img_folder_dir, seg_folder_dir) if seg_folder_dir else None
            img_filename = os.path.join(img_subfolder, os.path.basename(img_id))
            seg_filename = img_filename.replace(img_subfolder, seg_subfolder).replace(img_filetype, seg_filetype) if seg_subfolder else None
            img_data = tf.gfile.GFile(img_filename, "rb").read()
            img_height, img_width, img_channel = img_reader.read_image_dims(img_data)
            seg_data = tf.gfile.GFile(seg_filename, "rb").read()
            seg_height, seg_width, seg_channel = seg_reader.read_image_dims(seg_data)
            if not img_height == seg_height and img_width == seg_width:
                raise ValueError("not paired sizes of img and seg\n"
                                 "image: %s[%dx%dx%d], seg: %s[%dx%dx%d]"
                                 % (img_filename, img_height, img_width, img_channel, seg_filename,
                                    seg_height, seg_width, seg_channel))
            if min(img_height, img_width) < min_length:
                raise ValueError("smaller than min length: %s, h=%d, w=%d" % (img_filename, img_height, img_width))

            # Convert to tf example.
            file_id = os.path.basename(img_filename)
            example = build_data.image_n_seg_to_tfexample(img_data, file_id, img_height, img_width, seg_data)
            tfrecord_writer.write(example.SerializeToString())
            if (i + 1) % 100 == 0:
                print("Converting image %d/%d (shard %d) [%.4f seg/100]" % (i + 1, num_images, shard_id + 1, time.time() - t))
                t = time.time()
