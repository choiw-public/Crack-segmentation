"""
Folder structure should be like the below:

dataset/subsets/classes/img.jpg
ex) ImageNet/train/fish/fish1.jpg
    ImageNet/train/fish/fish2.jpg ...

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

FLAGS = tf.app.flags.FLAGS
raw_data_folder_dir = '/media/wooram/DATA/00.DL_datasets/ImageNet/2012/datasets/Raw_data/full_dataset/images'
tfrecord_dir = '/media/wooram/DATA/00.DL_datasets/ImageNet/2012/datasets/tfrecords/tfrecords_train_schedule/1to320/train'

label_min = 0
label_max = 320
set_name = 'train'  # specify subfolder names such as train, val, trainval etc.
num_shard = 320


def convert_dataset(filenames, gt_dict, set_name, num_shard):
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(num_shard)))

    image_reader = build_data.ImageReader('jpeg', channels=3)
    height_list = []
    width_list = []
    for shard_id in range(num_shard):
        output_filename = os.path.join(
            tfrecord_dir,
            '%s-%05d-of-%05d.tfrecord' % (set_name, shard_id + 1, num_shard))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)

            for i in range(start_idx, end_idx):
                # Read the image.
                image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                height, width, channel = image_reader.read_image_dims(image_data)
                height_list.append(float(height))
                width_list.append(float(width))
                clsname = filenames[i].split('/')[-2]
                label = gt_dict[clsname]
                # Convert to tf example.

                example = build_data.image_classification_to_tfexample(image_data, os.path.basename(filenames[i]),
                                                                       height,
                                                                       width,
                                                                       channel, int(label),
                                                                       clsname)
                tfrecord_writer.write(example.SerializeToString())
                if (i + 1) % 20.0 == 0:
                    print('>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id + 1))
    average_height = np.average(height_list)
    average_width = np.average(width_list)
    max_height = np.max(height_list)
    min_height = np.min(height_list)
    max_width = np.max(width_list)
    min_width = np.min(width_list)

    with open(os.path.join(tfrecord_dir, 'overall_info.txt'), 'w') as writer:
        writer.write('average_height: %.3f\n' % average_height)
        writer.write('average_width: %.3f\n' % average_width)
        writer.write('max_height: %.3f\n' % max_height)
        writer.write('min_height: %.3f\n' % min_height)
        writer.write('max_width: %.3f\n' % max_width)
        writer.write('min_width: %.3f\n' % min_width)
    writer.close()


def gen_class_and_label(List):
    new_List = []
    cls_n_lbl = {}
    classes = sorted(list(set([filename.split('/')[-2] for filename in List])))
    classes = classes[label_min:label_max]
    for i, cls in enumerate(classes):
        cls_n_lbl[cls] = i
    for entry in List:
        if entry.split('/')[-2] in classes:
            new_List.append(entry)
    return new_List, cls_n_lbl


def refine_list_with_only_images(List):
    allowed_extensions = ['jpg', 'JPG', 'png', 'PNG', 'JPEG']
    return [name for name in List if name.split('/')[-1].split('.')[-1] in allowed_extensions]


filelist = glob.glob(os.path.join(raw_data_folder_dir, set_name, '**/*'))
filelist, classes_n_labels = gen_class_and_label(filelist)
shuffle(filelist)
filelist = refine_list_with_only_images(filelist)
convert_dataset(filelist, classes_n_labels, set_name, num_shard)
