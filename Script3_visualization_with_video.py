from functions.project_fn.deploy_config import DeployConfig
from functions.project_fn.misc_utils import get_ckpt, list_getter
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

model_name = "model_last_continue"
video_dir = "./video_to_combine(389000)/raw"
tested_video_dir = "./video_to_combine(389000)"

config = DeployConfig(model_name, "vis")
ph_input = tf.placeholder(config.dtype, [None, None, None, 3])
data = {"input": ph_input, "gt": None}
ckpt = get_ckpt(config)
video_list = []
for ext in ["mp4", "mov"]:
    video_list += list_getter(video_dir, ext)
from functions.project_fn.model_utils import build_model


pred, hvd = build_model(data, config)
restorer = tf.train.Saver()
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
session_config.allow_soft_placement = True
session_config.gpu_options.visible_device_list = str(hvd.local_rank())
with tf.Session(config=session_config) as sess:

    print("Current ckpt: %s" % ckpt)
    restorer.restore(sess, ckpt)

    for video_name in video_list:
        video = cv.VideoCapture(video_name)
        should_continue, img = video.read()  # first frame
        if should_continue:
            fps = video.get(5)
            name_without_ext = os.path.basename(video_name)[:-4]
            dst_dir = os.path.dirname(video_name).replace(video_dir, tested_video_dir)
            superimposed_name = os.path.join(dst_dir, "superimposed", name_without_ext + ".avi")
            mask_name = os.path.join(dst_dir, "mask", name_without_ext + ".avi")
            h, w, _ = img.shape
            os.makedirs(os.path.dirname(superimposed_name), exist_ok=True)
            os.makedirs(os.path.dirname(mask_name), exist_ok=True)
            pred_np = sess.run(pred, {ph_input: np.reshape(img, (1, h, w, 3))})
            mask = np.ones_like(pred_np) - pred_np
            color_label = np.stack([np.zeros_like(pred_np), np.zeros_like(pred_np), pred_np * 255], 2)
            superimposed = img * np.expand_dims(mask, 2) + color_label
            superimposed_video = cv.VideoWriter(superimposed_name, cv.VideoWriter_fourcc(*"XVID"), fps, (w, h))
            mask_video = cv.VideoWriter(mask_name, cv.VideoWriter_fourcc(*"XVID"), fps, (w, h))
            superimposed_video.write(superimposed.astype(np.uint8))
            mask_video.write(color_label.astype(np.uint8))
        else:
            raise ValueError("the current video has no frame: %s" % video_name)

        while should_continue:
            should_continue, img = video.read()
            if should_continue:
                pred_np = sess.run(pred, {ph_input: np.reshape(img, (1, h, w, 3))})
                mask = np.ones_like(pred_np) - pred_np
                color_label = np.stack([np.zeros_like(pred_np), np.zeros_like(pred_np), pred_np * 255], 2)
                superimposed = img * np.expand_dims(mask, 2) + color_label
                superimposed_video.write(superimposed.astype(np.uint8))
                mask_video.write(color_label.astype(np.uint8))
        superimposed_video.release()
        mask_video.release()
