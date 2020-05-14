import os
import re
import cv2 as cv
import numpy as np
import tensorflow as tf
import time

raw_img_dir = "/media/wooram/DATA/00.DL_datasets/00.Civil ENG related dataset/deblur/1-0.pohang_sharp_longer_side_1920/images"
# sharp_img_dir and blur_img_dir are the output folder img_name
sharp_img_dir = "/media/wooram/DATA/00.DL_datasets/00.Civil ENG related dataset/deblur/2-0.pohang_synthetic_test_image_from_1-0/sharp"
blur_img_dir = "/media/wooram/DATA/00.DL_datasets/00.Civil ENG related dataset/deblur/2-0.pohang_synthetic_test_image_from_1-0/blur"
kernel_file = "../../kernels/gaussian5-31.tfrecord"
kernel_type = "gaussian"

num_cycles = 5
img_type = "color"


def list_getter(dir_name):
    image_list = []
    for path, subdirs, files in os.walk(dir_name):
        for name in files:
            if name.lower().endswith(("png", "jpg")):
                image_list.append(os.path.join(path, name))
    image_list.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])
    return image_list


def img_parser(blur_id):
    return {"img": tf.cast(tf.image.decode_png(tf.read_file(blur_id), channels=3), tf.float32), "filename": blur_id}


img_list = list_getter(raw_img_dir)
img_list = tf.convert_to_tensor(img_list)
data = tf.data.Dataset.from_tensor_slices(img_list)
data = data.apply(tf.data.experimental.map_and_batch(img_parser,
                                                     1,
                                                     drop_remainder=True)).make_initializable_iterator()
init = data.initializer
data = data.get_next()
tf_sharp = data["img"]
tf_file_id = data["filename"]

# kernel input pipeline
decode_features = {"kernel": tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                   "h": tf.FixedLenFeature([], tf.int64),
                   "w": tf.FixedLenFeature([], tf.int64)}


def blur_kernel_getter(tfrecord):
    def kernel_parser(entry):
        parsed = tf.parse_single_example(entry, decode_features)
        kernel_data = tf.convert_to_tensor(parsed["kernel"])
        h = tf.convert_to_tensor(parsed["h"])
        w = tf.convert_to_tensor(parsed["w"])
        return tf.reshape(kernel_data, [h, w])

    return tf.data.TFRecordDataset(tfrecord).repeat().shuffle(1000).map(kernel_parser).make_one_shot_iterator().get_next()


if img_type == "gray":
    tf_sharp = tf.image.rgb_to_grayscale(tf_sharp)
    channel = 1
elif img_type == "color":
    tf_sharp = tf_sharp
    channel = 3
else:
    raise ValueError("supported")

tf_kernel = blur_kernel_getter(kernel_file)
tf_kernel = tf.concat([tf.expand_dims(tf.expand_dims(tf_kernel, 2), 3)] * channel, 2)
tf_blur = tf.nn.depthwise_conv2d(tf_sharp, tf_kernel, [1, 1, 1, 1], "SAME")

tf_psnr = tf.image.psnr(tf_sharp, tf_blur, 255.0)
tf_ssim = tf.image.ssim(tf_sharp, tf_blur, 255.0)

with tf.Session() as sess:
    for cycle in range(num_cycles):
        sess.run(init)
        log_lines = []
        while True:
            try:
                t = time.time()
                file_id, sharp, blur, psnr, ssim, kernel = sess.run([tf_file_id, tf_sharp, tf_blur, tf_psnr, tf_ssim, tf_kernel])
                print(time.time() - t)
                kernel_shape = np.squeeze(kernel).shape[0]  # assume kernel has a squre size
                if kernel_shape % 2 == 0:
                    raise ValueError("kernel size must be odd number")
                cut_margin = int((kernel_shape - 1) / 2)
                sharp = np.squeeze(sharp).astype(np.uint8)[cut_margin:-cut_margin, cut_margin:-cut_margin]
                blur = np.squeeze(blur).astype(np.uint8)[cut_margin:-cut_margin, cut_margin:-cut_margin]

                base_folder = os.path.dirname(file_id[0])
                base_id = os.path.basename(file_id[0])

                sharp_folder = base_folder.replace(raw_img_dir, os.path.join(sharp_img_dir, kernel_type, "cycle_%02d" % cycle))
                blur_folder = base_folder.replace(raw_img_dir, os.path.join(blur_img_dir, kernel_type, "cycle_%02d" % cycle))
                if not os.path.exists(sharp_folder): os.makedirs(sharp_folder)
                if not os.path.exists(blur_folder): os.makedirs(blur_folder)
                sharp_fullpath = os.path.join(sharp_folder, base_id)
                blur_fullpath = os.path.join(blur_folder, base_id)
                if channel == 1:
                    cv.imwrite(sharp_fullpath, sharp)
                    cv.imwrite(blur_fullpath, blur)
                elif channel == 3:
                    cv.imwrite(sharp_fullpath, sharp[:, :, ::-1])
                    cv.imwrite(blur_fullpath, blur[:, :, ::-1])
                log_lines.append([base_id, psnr[0], ssim[0]])
            except tf.errors.OutOfRangeError:
                blur_img_log_dir = os.path.join(blur_img_dir, kernel_type, "cycle_%02d" % cycle)
                blur_img_log_name = os.path.join(blur_img_log_dir, "blur_image_profiles.csv")
                with open(blur_img_log_name, "w") as writer:
                    writer.write("file_id, blur_psnr, blur_ssim\n")
                    for log_line in log_lines:
                        writer.write("%s, %.6f, %.6f\n" % (log_line[0], log_line[1], log_line[2]))
                break
