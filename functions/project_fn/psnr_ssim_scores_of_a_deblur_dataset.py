import numpy as np
import tensorflow as tf
import os
import re
import multiprocessing
import matplotlib.pyplot as plt

from functions.project_fn.preprocess_developing import random_crop

# goprodataset
# tf_blur = "/media/wooram/DATA/00.DL_datasets/GOPRO_dataset_for_deblur/restructured/_train/images/tf_blur" # for tf_blur
# tf_blur = "/media/wooram/DATA/00.DL_datasets/GOPRO_dataset_for_deblur/restructured/_train/images/blur_gamma" # for blur_gamma
blur_dir = "/media/wooram/DATA/00.DL_datasets/GOPRO_dataset_for_deblur/restructured/_train/images/blur_gamma"
sharp_dir = "/media/wooram/DATA/00.DL_datasets/GOPRO_dataset_for_deblur/restructured/_train/images/sharp"
out_folder = "/media/wooram/DATA/00.DL_datasets/GOPRO_dataset_for_deblur/restructured/_train"
out_filename = "sharp_blur_gamma_psnr_ssim_of_goprodataset"
colormap_to_compare = "color"
do_crop = True


def get_image_list(dir_name):
    temp_list = []
    for path, subdirs, files in os.walk(dir_name):
        for name in files:
            if name.lower().endswith(("png", "jpg")):
                temp_list.append(os.path.join(path, name))
    temp_list.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])
    return temp_list


def input_from_image(image_list):
    # This function is intended to be used in inference.
    blur_list = image_list["blur"]
    sharp_list = image_list["sharp"]

    # check file extension
    extensions = []
    for blur_name, sharp_name in zip(blur_list, sharp_list):
        extensions.append(os.path.basename(blur_name).split(".")[-1])
        extensions.append(os.path.basename(sharp_name).split(".")[-1])
    extensions = list(set(extensions))
    if len(extensions) > 1:
        raise ValueError("Standardization of extensions is required. Easy words: use same image format (jpg, jpeg, png) for all images")
    elif len(extensions) == 0:
        raise ValueError("no image files exist")

    # choose correct image_decoder
    if extensions[0].lower() in ["jpg", "jpeg"]:
        img_decoder = tf.image.decode_jpeg
    elif extensions[0].lower() in ["png"]:
        img_decoder = tf.image.decode_png

    def parse_fn(blur_name, sharp_name):
        blur_img = img_decoder(tf.read_file(blur_name), channels=3)
        sharp_img = img_decoder(tf.read_file(sharp_name), channels=3)
        if colormap_to_compare == "gray":
            blur_img = tf.image.rgb_to_grayscale(blur_img)
            sharp_img = tf.image.rgb_to_grayscale(sharp_img)
        if do_crop:
            blur_img, sharp_img = random_crop([blur_img, sharp_img], 513, 513)
        return blur_img, sharp_img, blur_name

    blur_list = tf.convert_to_tensor(image_list["blur"], dtype=tf.string)
    sharp_list = tf.convert_to_tensor(image_list["sharp"], dtype=tf.string)

    blur_img_names = tf.data.Dataset.from_tensor_slices(blur_list)
    sharp_img_name = tf.data.Dataset.from_tensor_slices(sharp_list)
    img_names = tf.data.Dataset.zip((blur_img_names, sharp_img_name))

    workers = multiprocessing.cpu_count() / 2
    data_iterator = img_names.map(parse_fn, num_parallel_calls=workers).batch(1).make_initializable_iterator()
    data = data_iterator.get_next()
    return data[0], data[1], data[2], data_iterator.initializer


# tf_blur vs sharp
img_list = dict()
img_list["blur"] = get_image_list(blur_dir)
img_list["sharp"] = get_image_list(sharp_dir)

tf_blur, tf_sharp, tf_filename, data_init = input_from_image(img_list)
tf_psnr = tf.image.psnr(tf_blur, tf_sharp, 255)
tf_ssim = tf.image.ssim(tf_blur, tf_sharp, 255)

with tf.Session() as sess:
    sess.run(data_init)
    out_filename = os.path.join(out_folder, out_filename + "_" + colormap_to_compare)
    if do_crop:
        out_filename = out_filename + "_cropped"
    with open(out_filename, "w") as writer:
        writer.write("filename, psnr, ssim\n")
        while True:
            try:
                psnr, ssim, filename, blur, sharp = sess.run([tf_psnr, tf_ssim, tf_filename, tf_blur, tf_sharp])
                filename = filename[0].replace(blur_dir, "")
                writer.write("%s, %.4f, %.4f\n" % (filename, psnr, ssim))
                print
            except:
                break
