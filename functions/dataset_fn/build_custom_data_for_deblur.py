import re
import math
import os.path
import build_data
import tensorflow as tf
from random import shuffle

FLAGS = tf.app.flags.FLAGS
sharp_img_folder_dir = "/media/choiw/DATA/00.DL_datasets/00.civil/deblur/00.train/1-1.pohang_sharp_longer_side_1920_x256_aug/images"
blur_img_folder_dir = None
tfrecord_out_dir = "/media/choiw/DATA/00.DL_datasets/00.civil/deblur/00.train/1-1.pohang_sharp_longer_side_1920_x256_aug/tfrecord"  # folder path or none
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


sharp_img_list = get_img_list(sharp_img_folder_dir)
if not sharp_img_list:
    raise ValueError("no images exist in the specified folder")
if blur_img_folder_dir:
    blur_img_list = get_img_list(blur_img_folder_dir)
    if not blur_img_list:
        raise ValueError("no images exist in the specified folder")
else:
    blur_img_list = None

if blur_img_list:
    # if tf_blur images are given
    # inspect file names of sharp images and tf_blur images
    if not len(sharp_img_list) == len(blur_img_list):
        raise ValueError("number of sharp images and tf_blur images are different")
    for sharp_name, blur_name in zip(sharp_img_list, blur_img_list):
        sharp_name = sharp_name.replace(sharp_img_folder_dir, "")
        blur_name = blur_name.replace(blur_img_folder_dir, "")
        if not sharp_name == blur_name:
            raise ValueError("sharp image img_name and tf_blur image img_name is not paired:\n"
                             "sharp image img_name: %s, tf_blur image img_name: %s" % (sharp_name, blur_name))

sharp_filetype = get_unique_file_extension(sharp_img_list)

if blur_img_list:
    blur_filetype = get_unique_file_extension(blur_img_list)
else:
    blur_filetype = None

shuffle(sharp_img_list)

num_images = len(sharp_img_list)
num_per_shard = int(math.ceil(num_images / float(num_shards)))

sharp_img_reader = build_data.ImageReader(sharp_filetype, channels=3)
if blur_img_list:
    blur_img_reader = build_data.ImageReader(blur_filetype, channels=3)
else:
    blur_img_reader = None
if not os.path.exists(tfrecord_out_dir):
    os.makedirs(tfrecord_out_dir)

for shard_id in range(num_shards):
    output_filename = os.path.join(tfrecord_out_dir, "%s-%05d-of-%05d.tfrecord" % (dataset_split_name, shard_id + 1, num_shards))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        start_idx = shard_id * num_per_shard
        end_idx = min((shard_id + 1) * num_per_shard, num_images)
        for i in range(start_idx, end_idx):
            if i % 500 == 0:
                print("Converting image %d/%d shard %d" % (i + 1, num_images, shard_id + 1))
            # Read the image.
            sharp_subfolder = os.path.dirname(sharp_img_list[i])
            blur_subfolder = sharp_subfolder.replace(sharp_img_folder_dir, blur_img_folder_dir) if blur_img_folder_dir else None
            sharp_filename = os.path.join(sharp_subfolder, os.path.basename(sharp_img_list[i]))
            blur_filename = sharp_filename.replace(sharp_subfolder, blur_subfolder).replace(sharp_filetype, blur_filetype) if blur_subfolder else None
            sharp_img_data = tf.gfile.GFile(sharp_filename, "r").read()
            sharp_height, sharp_width, sharp_channel = sharp_img_reader.read_image_dims(sharp_img_data)
            if blur_filename:
                blur_img_data = tf.gfile.GFile(blur_filename, "r").read()
                blur_height, blur_width, blur_channel = blur_img_reader.read_image_dims(blur_img_data)
                if not sharp_height == blur_height and sharp_width == blur_width and sharp_channel == blur_channel:
                    raise ValueError("sharp and tf_blur image dimensions are different\n"
                                     "sharp: %s[%dx%dx%d], tf_blur: %s[%dx%dx%d]"
                                     % (sharp_filename, sharp_height, sharp_width, sharp_channel, blur_filename,
                                        blur_height, blur_width, blur_channel))

            # Convert to tf example.
            file_id = os.path.basename(sharp_filename)
            if blur_img_list:
                example = build_data.sharp_n_blur_images_to_tfexample(sharp_img_data, blur_img_data, file_id, sharp_height, sharp_width)
            else:
                example = build_data.sharp_images_to_tfexample(sharp_img_data, file_id, sharp_height, sharp_width)
            tfrecord_writer.write(example.SerializeToString())
