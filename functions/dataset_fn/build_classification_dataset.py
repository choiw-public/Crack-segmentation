"""
Folder structure should be like the below:

dataset/subsets/classes/img.jpg
ex) ImageNet/_train/fish/fish1.jpg
    ImageNet/_train/fish/fish2.jpg ...

    ImageNet/val/animal/animal1.jpg
    ImageNet/val/animal/animal2.jpg

Each file img_name must include classe label
"""
from random import shuffle
import glob
import math
import os.path
import build_data
import tensorflow as tf
import numpy as np
from collections import OrderedDict
import multiprocessing
from joblib import Parallel, delayed

FLAGS = tf.app.flags.FLAGS
src_dir = "../../datasets/cat_n_dog/_train"
tfrecord_dir = "../../datasets/cat_n_dog/tfrecord"

set_name = "_train"  # specify subfolder names such as _train, val, trainval etc.
num_shard = 10


def convert_dataset(filenames, gt_dict, shard_idx):
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(num_shard)))

    image_reader = build_data.ImageReader("jpeg", channels=3)
    height_list = []
    width_list = []
    for shard_id in range(num_shard):
        output_filename = os.path.join(tfrecord_dir, "%s-%05d-of-%05d.tfrecord" % (set_name, shard_idx, num_shard))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)

            for i in range(start_idx, end_idx):
                # Read the image.
                image_data = tf.gfile.FastGFile(filenames[i], "r").read()
                height, width, channel = image_reader.read_image_dims(image_data)
                height_list.append(float(height))
                width_list.append(float(width))
                clsname = filenames[i].split("/")[-2]
                label = gt_dict[clsname]
                # Convert to tf example.

                example = build_data.image_classification_to_tfexample(image_data,
                                                                       os.path.basename(filenames[i]),
                                                                       height,
                                                                       width,
                                                                       int(label),
                                                                       clsname)
                tfrecord_writer.write(example.SerializeToString())
                if (i + 1) % 10 == 0:
                    print(">> Converting image %d/%d shard %d" % (i + 1, len(filenames), shard_id + 1))


def gen_class_and_label(file_list):
    cls_n_lbl = OrderedDict()
    classes = sorted(list(set([filename.split("/")[-2] for filename in file_list])))
    for i, cls in enumerate(classes):
        cls_n_lbl[cls] = i
    return cls_n_lbl


def refine_list_with_only_images(filelist):
    allowed_extensions = ["jpg", "JPG", "png", "PNG", "JPEG"]
    return [name for name in filelist if name.split("/")[-1].split(".")[-1] in allowed_extensions]


def divide_list(filelist, division):
    num = int(math.ceil(float(len(filelist)) / division))
    new_filelist = []
    while True:
        try:
            tmp_list = []
            for i in range(num):
                tmp_list.append(filelist.pop(0))
            new_filelist.append(tmp_list)
        except:
            new_filelist.append(tmp_list)
            break
    return new_filelist


filelist_full = glob.glob(os.path.join(src_dir, "**/*"))
shuffle(filelist_full)
filelist_full = refine_list_with_only_images(filelist_full)
classes_n_labels = gen_class_and_label(filelist_full)

with open(os.path.join(tfrecord_dir, "class_info.csv"), "w") as writer:
    writer.write("name,label\n")
    for k, v in classes_n_labels.iteritems():
        writer.write("%s, %d\n" % (k, v))

filelist_full = divide_list(filelist_full, num_shard)
Parallel(n_jobs=2)(delayed(convert_dataset)(sublist, classes_n_labels, idx) for idx, sublist in enumerate(filelist_full))
