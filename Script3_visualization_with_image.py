from functions.project_fn.model_utils import build_model
from functions.project_fn.input_pipeline import get_image_list, build_input_pipeline

from functions.project_fn.deploy_config import DeployConfig
from functions.project_fn.misc_utils import get_ckpt, list_getter
from functions.project_fn.metric_calculation import log_initialize, write_eval_log, calculate_segmentation_metric
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time

model_name = "fullmodel1_continue"

config = DeployConfig(model_name, "vis")
img_ph = tf.placeholder(config.dtype, [None, None, None, 3])
data = {"input": img_ph, "gt": None}
pred, hvd = build_model(data, config)
restorer = tf.train.Saver()
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
session_config.allow_soft_placement = True
session_config.gpu_options.visible_device_list = str(hvd.local_rank())
with tf.Session(config=session_config) as sess:
    ckpt = get_ckpt(config)
    print("Current ckpt: %s" % ckpt)
    restorer.restore(sess, ckpt)
    img_list = list_getter(config.img_dir, extension="jpg")
    for img_name in img_list:
        img = cv.imread(img_name)[:, :, ::-1]
        if config.dtype == tf.float16:
            img_to_feed = img.astype(np.float16)
        elif config.dtype == tf.float32:
            img_to_feed = img.astype(np.float32)
        pred_np = sess.run(pred, {img_ph: np.expand_dims(img_to_feed, 0)})
        name_without_ext = os.path.basename(img_name)[:-4]
        base_folder = os.path.dirname(img_name).replace(config.img_dir, config.vis_result_dir)
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
        mask_name = os.path.join(base_folder, name_without_ext + "_mask.png")
        superimposed_name = os.path.join(base_folder, name_without_ext + "_superimposed.jpg")
        print("current image name: %s" % os.path.basename(name_without_ext))

        mask = pred_np.squeeze().astype(np.uint8)[:, :, ::-1]
        superimposed = cv.addWeighted(img[:, :, ::-1], 1.0, mask, 1.0, 0)
        try:
            cv.imwrite(mask_name, mask)
            cv.imwrite(superimposed_name, superimposed)
        except:
            print("debug from here")

