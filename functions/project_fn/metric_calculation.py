import tensorflow as tf
import csv

from functions.project_fn.misc_utils import get_tensor_shape
from collections import OrderedDict
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
from joblib import Parallel, delayed
from functions.project_fn import misc_utils
import multiprocessing


# colormap = np.load("./colormap.npy")


def get_metric_head(config):
    metric_head = []
    for metric_id in ["accuracy"]:
        for class_id in range(config.num_of_classes):
            metric_head.append("C" + str(class_id) + "_" + metric_id)
    metric_head.append("mIoU")
    return metric_head


def get_tensors_for_evaluation(tf_result, tf_data_gt, config):
    if config.task == "deblur":
        tf_psnr_deblur = tf.squeeze(tf.image.psnr(tf_data_gt, tf_result, 255))
        tf_ssim_deblur = tf.squeeze(tf.image.ssim(tf_data_gt, tf_result, 255))
        return [tf_psnr_deblur, tf_ssim_deblur]
    elif config.task in ["segmentation", "deblur-segmentation"]:
        return tf.confusion_matrix(tf.reshape(tf_data_gt, [-1]), tf.reshape(tf_result, [-1]), config.num_of_classes, dtype=tf.float32)
    else:
        raise ValueError("not supported")


def log_initialize(config):
    with open(os.path.join(config.eval_log_dir, "00.metric_overall.csv"), "a+") as writer:
        writer.seek(0)  # python 3, this line must be included. it"s a python bug.
        log = writer.readlines()
        if not log:
            if config.num_classes <= 2:
                writer.write("ckpt_id, precision, recall, f1, miou\n")
            else:
                writer.write("ckpt_id, miou\n")
        else:
            log = [entry.strip() for entry in log]
    return log


def create_or_read_existing_log2(config):
    if "train" in config.img_dir:
        prefix = "train"
    elif "test" in config.img_dir:
        prefix = "test"
    else:
        prefix = ""
    with open(os.path.join(config.eval_log_dir, "00.metric_overall_%s.csv" % prefix), "a+") as writer:
        log = writer.readlines()
        if not log:
            writer.write("ckpt_id, accuracy\n")
        else:
            log = [entry.strip() for entry in log]
    return log


def calculate_segmentation_metric(cmatrix):
    tp = np.diag(cmatrix)
    fp = np.sum(cmatrix, axis=0) - tp
    fn = np.sum(cmatrix, axis=1) - tp
    precision = tp / (tp + fp)  # precision of each class. [batch, class]
    recall = tp / (tp + fn)  # recall of each class. [batch, class]
    f1 = 2 * precision * recall / (precision + recall)
    iou = tp / (tp + fp + fn)  # iou of each class. [batch, class]
    miou = iou.mean()  # miou
    if iou.shape[0] <= 2:
        return [precision[1], recall[1], f1[1], miou]
    else:
        return [miou]


def inf_or_nan_to_zero(value):
    if np.isnan(value):
        return 0
    elif np.isinf(value):
        return 0
    else:
        return value


def write_eval_log(ckpt_id, metrics, config):
    with open(os.path.join(config.eval_log_dir, "00.metric_overall.csv"), "a+") as writer:
        writer.write("%s, " % ckpt_id)
        writer.write(", ".join([str(value) for value in metrics]) + "\n")


def start_eval(ckpt_id, tf_filename, tf_eval_tensors, sess, config, tf_sharp, tf_blur, tf_deblur):
    image_ids, per_image_metric_record = [], []
    if config.task in ["segmentation", "deblur-segmentation"]:
        cumulative_cmatrix = np.zeros((config.num_of_classes, config.num_of_classes))
    else:
        cumulative_cmatrix = None
    num_tested_img = 0
    while True:
        try:
            t = time.time()
            # eval_tensor is confucion matrix if config.task in ["segmentation","deblur-segmentation"]
            # eval_tensor is PSNR and SSIM scores if config.task is "deblur"
            # filename, eval_tensor = sess.run([tf_filename, tf_eval_tensors])
            filename, eval_tensor, sharp, blur, deblur = sess.run([tf_filename, tf_eval_tensors, tf_sharp, tf_blur, tf_deblur])
            filename = filename[0]
            num_tested_img += 1
            image_id = os.path.basename(filename)
            if num_tested_img % 10 == 0:
                print("file id: %s [%.4f sec/img]" % (image_id, time.time() - t))
            image_ids.append(image_id)
            if config.task in ["segmentation", "deblur-segmentation"]:
                cumulative_cmatrix += eval_tensor
                per_image_metric_record.append(calculate_segmentation_metric(eval_tensor, config))
            else:
                per_image_metric_record.append(eval_tensor)
        except tf.errors.OutOfRangeError:
            if config.task in ["segmentation", "deblur-segmentation"]:
                overall_metric = calculate_segmentation_metric(cumulative_cmatrix, config)
            elif config.task == "deblur":
                overall_metric = np.average(np.array(per_image_metric_record), 0)
            else:
                raise ValueError("not supported")
            write_eval_log(ckpt_id, image_ids, per_image_metric_record, overall_metric, config)
            break
    if config.task == "deblur":
        misc_utils.categorized_ssim(config)


def start_eval_with_thresh(ckpt_id, thresh, tf_filename, tf_logit, tf_gt, sess, config):
    if config.task not in ["segmentation", "deblur-segmentation"]:
        raise ValueError("this function is only for binary segmenation")

    num_tested_img = 0
    total_tp, total_tn, total_fp, total_fn = 0.0, 0.0, 0.0, 0.0
    tp_record, tn_record, fp_record, fn_record = [], [], [], []
    image_ids, per_image_metric_record = [], []
    while True:
        try:
            t = time.time()
            logit, gt, filename = sess.run([tf_logit, tf_gt, tf_filename])
            filename = filename[0]
            num_tested_img += 1
            image_id = os.path.basename(filename).split(".")[0]
            pred = (logit[:, :, :, 1] >= thresh).astype(np.float32)

            gt = np.squeeze(gt)
            tp = np.sum(np.logical_and(pred == 1, gt == 1)).astype(np.float32)
            tn = np.sum(np.logical_and(pred == 0, gt == 0)).astype(np.float32)
            fp = np.sum(np.logical_and(pred == 1, gt == 0)).astype(np.float32)
            fn = np.sum(np.logical_and(pred == 0, gt == 1)).astype(np.float32)

            # consider label1 as positive
            precision_c1 = tp / (tp + fp)
            recall_c1 = tp / (tp + fn)
            f1_c1 = 2 * precision_c1 * recall_c1 / (precision_c1 + recall_c1)
            iou_c1 = tp / (tp + fp + fn)

            # consider label0 as positive
            precision_c0 = tn / (tn + fn)
            recall_c0 = tn / (tn + fp)
            f1_c0 = 2 * precision_c0 * recall_c0 / (precision_c0 + recall_c0)
            iou_c0 = tn / (tn + fp + fn)

            miou = (iou_c0 + iou_c1) / 2

            per_image_metric_record.append([tn, tp, fn, fp, fp, fn, precision_c0, precision_c1, recall_c0, recall_c1, f1_c0, f1_c1, iou_c0, iou_c1, miou])

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

            tp_record.append(tp)
            tn_record.append(tn)
            fp_record.append(fp)
            fn_record.append(fn)

            if num_tested_img % 10 == 0:
                print("file id: %s [%.4f sec/img]" % (image_id, time.time() - t))
            image_ids.append(image_id)

        except tf.errors.OutOfRangeError:
            if config.task not in ["segmentation", "deblur-segmentation"]:
                raise ValueError("this function is only for binary segmenation")
            per_image_metric_record = np.array(per_image_metric_record).astype(np.float32)
            overall_metric = per_image_metric_record[:, 0:6]
            overall_metric = np.sum(overall_metric, axis=0)
            overall_tp = overall_metric[1]
            overall_tn = overall_metric[0]
            overall_fp = overall_metric[3]
            overall_fn = overall_metric[2]

            # consider label1 as positive
            precision_c1 = overall_tp / (overall_tp + overall_fp)
            recall_c1 = overall_tp / (overall_tp + overall_fn)
            f1_c1 = 2 * precision_c1 * recall_c1 / (precision_c1 + recall_c1)
            iou_c1 = overall_tp / (overall_tp + overall_fp + overall_fn)

            # consider label0 as positive
            precision_c0 = overall_tn / (overall_tn + overall_fn)
            recall_c0 = overall_tn / (overall_tn + overall_fp)
            f1_c0 = 2 * precision_c0 * recall_c0 / (precision_c0 + recall_c0)
            iou_c0 = overall_tn / (overall_tn + overall_fp + overall_fn)

            miou = (iou_c0 + iou_c1) / 2

            overall_metric = list(overall_metric) + [precision_c0, precision_c1, recall_c0, recall_c1, f1_c0, f1_c1, iou_c0, iou_c1, miou]
            write_eval_log(ckpt_id, image_ids, per_image_metric_record, overall_metric, config)
            break


def start_thresh_analysis(ckpt_id, tf_logits, tf_gt, sess, config):
    def thresh_analysis(_logit, _label, _thresh):
        _prediction = (_logit >= _thresh).astype(np.float32)
        _TP = np.sum(np.logical_and(_prediction == 1, _label == 1))
        _TN = np.sum(np.logical_and(_prediction == 0, _label == 0))
        _FP = np.sum(np.logical_and(_prediction == 1, _label == 0))
        _FN = np.sum(np.logical_and(_prediction == 0, _label == 1))
        return {"TP": _TP, "TN": _TN, "FP": _FP, "FN": _FN}

    path = os.path.join(config.eval_log_dir, "thresh_hold_analysis")
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, "00.thresh_analysis_overall.csv"), "a+") as writer:
        log = writer.readlines()
        if not log:
            writer.write("ckpt_id, thresh hold, ")
            writer.write(", ".join(get_metric_head(config)) + "\n")

    thresh_holds = np.linspace(0.0, 1.0, 2000)
    total_logit = []
    total_gt = []
    with open(os.path.join(path, "thresh_analysis(%s).csv" % ckpt_id), "w") as writer:
        writer.write("thresh hold, ")
        writer.write(", ".join(get_metric_head(config)) + "\n")
    num_tested_img = 0
    while True:
        try:
            logit, gt = sess.run([tf_logits, tf_gt])
            num_tested_img += 1
            total_logit.append(np.squeeze(logit[:, :, :, 1]))
            total_gt.append(np.squeeze(gt))

        except tf.errors.OutOfRangeError:
            per_image_metric_record = []
            for thresh in thresh_holds:
                results = Parallel(n_jobs=10)(delayed(thresh_analysis)(logit, label, thresh) for logit, label in zip(total_logit, total_gt))
                tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
                for entries in results:
                    tp += entries["TP"]
                    tn += entries["TN"]
                    fp += entries["FP"]
                    fn += entries["FN"]

                if tp != 0.0 and tn != 0.0:
                    # consider label1 as positive
                    precision_c1 = tp / (tp + fp)
                    recall_c1 = tp / (tp + fn)
                    f1_c1 = 2 * precision_c1 * recall_c1 / (precision_c1 + recall_c1)
                    iou_c1 = tp / (tp + fp + fn)

                    # consider label0 as positive
                    precision_c0 = tn / (tn + fn)
                    recall_c0 = tn / (tn + fp)
                    f1_c0 = 2 * precision_c0 * recall_c0 / (precision_c0 + recall_c0)
                    iou_c0 = tn / (tn + fp + fn)

                    miou = (iou_c0 + iou_c1) / 2
                    overall_metric = [tn, tp, fn, fp, fp, fn, precision_c0, precision_c1, recall_c0, recall_c1, f1_c0, f1_c1, iou_c0, iou_c1, miou]
                    per_image_metric_record.append(overall_metric)
                    with open(os.path.join(path, "thresh_analysis(%s).csv" % ckpt_id), "a+") as writer:
                        writer.write(str(thresh) + ",")
                        writer.write(", ".join([str(value) for value in overall_metric]) + "\n")
            per_image_metric_record = np.array(per_image_metric_record)
            idx = np.where(per_image_metric_record[:, 11] == max(per_image_metric_record[:, 11]))[0][0]
            best_metric = per_image_metric_record[idx, :]
            best_thresh = thresh_holds[idx]
            with open(os.path.join(path, "00.thresh_analysis_overall.csv"), "a+") as writer:
                writer.write(ckpt_id + ",")
                writer.write(str(best_thresh) + ",")
                writer.write(", ".join([str(value) for value in best_metric]) + "\n")
            break


# def start_vis(tf_filename, tf_result, tf_input, tf_gt, sess, config, tpfnfp=False):
def start_vis(tf_filename, tf_result, sess, config, tpfnfp=False):
    num_tested_img = 0
    tf_result = tf.squeeze(tf_result)
    if config.task in ["segmentation", "deblur-segmentation"]:
        tf_result = tf.one_hot(tf_result, config.num_of_classes)

    save_folder = os.path.join(config.vis_result_dir, config.task)
    overlay_folder = os.path.join(config.vis_result_dir, "overlay")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(overlay_folder):
        os.makedirs(overlay_folder)

    def change_colormap_tpfnfp(prediction, label):
        # only valid if binary segmentation
        prediction = prediction.astype(np.float)
        prediction[prediction > 0] = 1
        label = label.astype(np.float)
        label[label > 0] = 1
        new_prediction = np.zeros_like(prediction)

        new_prediction[prediction + label == 2] = 1  # tp
        new_prediction[prediction - label == -1] = 2  # fn
        new_prediction[label - prediction == -1] = 3  # fp

        return new_prediction.astype(np.uint8)

    # colormap = np.load("./colormap.npy")
    colormap = np.array([[0, 0, 0],  # class 0, background
                         [255, 0, 0],  # class 1, R
                         [0, 255, 0],  # class 2, G
                         [0, 0, 255]])  # class 3, B
    while True:
        try:
            # filename, result, img, gt = sess.run([tf_filename, tf_result, tf_input, tf_gt])
            if config.task in ["segmentation", "deblur-segmentation"]:
                t = time.time()
                filename, result, img, gt = sess.run([tf_filename, tf_result])
                img = np.squeeze(img).astype(np.uint8)
                gt = np.squeeze(gt).astype(np.uint8)
                elapsed = time.time() - t
                filename = filename[0]
                num_tested_img += 1
                image_id = os.path.basename(filename)
                seg_filename = os.path.join(save_folder, image_id)
                overlay_filename = os.path.join(overlay_folder, image_id)
            else:
                t = time.time()
                filename, result = sess.run([tf_filename, tf_result])
                elapsed = time.time() - t
                image_id = os.path.basename(filename[0])
                deblur_filename = os.path.join(save_folder, image_id)

            if config.task in ["segmentation", "deblur-segmentation"]:
                prediction = np.argmax(result, len(result.shape) - 1)
                if tpfnfp:
                    prediction = change_colormap_tpfnfp(prediction, gt)
                prediction_vis = colormap[prediction][:, :, ::-1]
                overlayed = cv.addWeighted(img[:, :, ::-1], 0.6, prediction_vis.astype(np.uint8), 1.0, 0)
                cv.imwrite(overlay_filename, overlayed)
                prediction_vis = cv.cvtColor(prediction_vis.astype(np.uint8), cv.COLOR_BGR2BGRA)
                prediction_vis[np.all(prediction_vis == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]
                cv.imwrite(seg_filename, prediction_vis.astype(np.uint8))

            elif config.task == "deblur":
                cv.imwrite(deblur_filename, result.astype(np.uint8)[:, :, ::-1])
            if num_tested_img % 10 == 0:
                print("file id: %s [%.4f sec/img]" % (image_id, elapsed))
        except tf.errors.OutOfRangeError:
            break


def start_vis_with_thresh(tf_filename, thresh, tf_result, tf_input, tf_gt, sess, config, tpfnfp=False):
    num_tested_img = 0
    seg_folder = os.path.join(config.vis_result_dir, "seg")
    overlay_folder = os.path.join(config.vis_result_dir, "overlay")
    if not os.path.exists(seg_folder):
        os.makedirs(seg_folder)
    if not os.path.exists(overlay_folder):
        os.makedirs(overlay_folder)

    def change_colormap_tpfnfp(prediction, label):
        # only valid if binary segmentation
        prediction = prediction.astype(np.float)
        prediction[prediction > 0] = 1
        label = label.astype(np.float)
        label[label > 0] = 1
        new_prediction = np.zeros_like(prediction)

        new_prediction[prediction + label == 2] = 1  # tp
        new_prediction[prediction - label == -1] = 2  # fn
        new_prediction[label - prediction == -1] = 3  # fp

        return new_prediction.astype(np.uint8)

    # colormap = np.load("./colormap.npy")
    colormap = np.array([[0, 0, 0],  # class 0, background
                         [255, 0, 0],  # class 1, R
                         [0, 255, 0],  # class 2, G
                         [0, 0, 255]])  # class 3, B
    while True:
        try:
            t = time.time()
            filename, logit, img, gt = sess.run([tf_filename, tf_result, tf_input, tf_gt])
            img = np.squeeze(img).astype(np.uint8)
            gt = np.squeeze(gt).astype(np.uint8)
            elapsed = time.time() - t
            filename = filename[0]
            num_tested_img += 1
            image_id = os.path.basename(filename).split(".")[0]
            seg_filename = os.path.join(seg_folder, image_id) + ".png"
            overlay_filename = os.path.join(overlay_folder, image_id) + ".png"
            # todo: the below if statements may give hint for debugging this script.

            if config.task not in ["segmentation", "deblur-segmentation"]:
                raise ValueError("this function is only for binary segmenation")
            prediction = np.squeeze((logit[:, :, :, 1] >= thresh).astype(np.float32))
            if tpfnfp:
                prediction = change_colormap_tpfnfp(prediction, gt)
            prediction_vis = colormap[prediction][:, :, ::-1]
            overlayed = cv.addWeighted(img[:, :, ::-1], 0.6, prediction_vis.astype(np.uint8), 1.0, 0)
            cv.imwrite(overlay_filename, overlayed)
            prediction_vis = cv.cvtColor(prediction_vis.astype(np.uint8), cv.COLOR_BGR2BGRA)
            prediction_vis[np.all(prediction_vis == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]
            cv.imwrite(seg_filename, prediction_vis.astype(np.uint8))

            if num_tested_img % 10 == 0:
                print("file id: %s [%.4f sec/img]" % (image_id, elapsed))
        except tf.errors.OutOfRangeError:
            break


def start_filter_out_crack_img(tf_filename, thresh, tf_result, tf_input, tf_gt, sess, config, tpfnfp=False):
    num_tested_img = 0
    while True:
        try:
            filename, logit, img, gt = sess.run([tf_filename, tf_result, tf_input, tf_gt])
            img = np.squeeze(img).astype(np.uint8)
            # todo: the below if statements may give hint for debugging this script.

            if config.task not in ["segmentation", "deblur-segmentation"]:
                raise ValueError("this function is only for binary segmenation")
            prediction = np.squeeze((logit[:, :, :, 1] >= thresh).astype(np.float32))
            if np.sum(prediction) > 20:
                num_tested_img += 1
                new_filename = "./tmp/maybe_crack%05d.png" % num_tested_img
                cv.imwrite(new_filename, img[:, :, ::-1])
                if num_tested_img % 200 == 0:
                    print(num_tested_img)
        except tf.errors.OutOfRangeError:
            break
