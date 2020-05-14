from functions.project_fn import misc_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def miou_loss(logit, ground_truth, config):
    # calculated bache mean intersection over union loss
    if ground_truth is None:
        raise ValueError("ground_truth cannot be None")
    if config.dtype == tf.float16:
        logit = tf.cast(logit, tf.float32)
    prob_map = tf.nn.softmax(logit)
    onehot_gt = tf.one_hot(tf.cast(tf.squeeze(ground_truth, 3), tf.uint8), config.num_classes)

    logit_shape = misc_utils.get_tensor_shape(prob_map)
    # gt_shape = get_tensor_shape(onehot_gt)

    # check sanity
    if config.num_classes == 1:
        raise ValueError("Number of classes is at least 2, background class must be included.")
    # if logit_shape != gt_shape:
    #     raise ValueError("Mismatching shapes between logits and ground_truths")
    if logit_shape[-1] != config.num_classes:
        raise ValueError("Something wrong. Check the get_shape of logit, ground_truth, and config.num_of_classes")

    # for calculating iou loss with logit
    intersection_logit = prob_map * onehot_gt  # [batch, height, width, class]
    union_logit = prob_map + onehot_gt - intersection_logit  # [batch, height, width, class]
    iou_logit = tf.reduce_sum(intersection_logit, [0, 1, 2]) / tf.reduce_sum(union_logit, [0, 1, 2])  # class
    miou_logit = tf.reduce_mean(iou_logit)
    loss = 1.0 - tf.reduce_mean(miou_logit)
    tf.add_to_collection("miou_loss", loss)
    return loss


def focal_loss(logit, gt, config, gamma=2.0):
    pred = tf.nn.softmax(logit)  # [batch_size,num_classes]
    onehot_gt = tf.one_hot(tf.cast(tf.squeeze(gt, 3), tf.int32), config.num_classes)
    loss = tf.reduce_sum(-onehot_gt * ((1 - pred) ** gamma) * tf.log(pred))
    # loss = tf.reduce_mean(tf.reduce_sum(-onehot_gt * ((1 - pred) ** gamma) * tf.log(pred), [1, 2, 3]))
    tf.add_to_collection("focal_loss", loss)
    return loss


def focal_miou_loss(logit, gt, config):
    return 0.5 * miou_loss(logit, gt, config) + 0.5 * focal_loss(logit, gt, config)


def weighted_miou_loss(logit, ground_truth, config):
    class_weight = config.loss_fn_names[1]  # [background crack]
    # calculated bache mean intersection over union loss
    if ground_truth is None:
        raise ValueError("ground_truth cannot be None")
    onehot_gt = tf.one_hot(tf.cast(tf.squeeze(ground_truth, 3), tf.int32), config.num_of_classes)

    logit_shape = misc_utils.get_tensor_shape(logit)
    # gt_shape = get_tensor_shape(onehot_gt)

    # check sanity
    if config.num_of_classes == 1:
        raise ValueError("Number of classes is at least 2, background class must be included.")
    # if logit_shape != gt_shape:
    #     raise ValueError("Mismatching shapes between logits and ground_truths")
    if logit_shape[-1] != config.num_of_classes:
        raise ValueError("Something wrong. Check the get_shape of logit, ground_truth, and config.num_of_classes")

    # for calculating iou loss with logit
    intersection_logit = logit * onehot_gt  # [batch, height, width, class]
    union_logit = logit + onehot_gt - intersection_logit  # [batch, height, width, class]
    iou_logit = tf.reduce_sum(intersection_logit, [0, 1, 2]) / tf.reduce_sum(union_logit, [0, 1, 2])  # class
    loss = 1.0 - (iou_logit[0] * class_weight[0] + iou_logit[1] * class_weight[1])
    tf.add_to_collection("onehot_gt", onehot_gt)
    tf.add_to_collection("logit", logit)

    # for monitoring purpose
    tf.add_to_collection("intersection_logit", intersection_logit)
    tf.add_to_collection("union_logit", union_logit)
    return loss


def mf1_loss(logit, ground_truth, config):
    # calculated f1 loss
    if ground_truth is None:
        raise ValueError("ground_truth cannot be None")
    onehot_gt = tf.one_hot(tf.cast(tf.squeeze(ground_truth, 3), tf.int32), config.num_of_classes)

    logit_shape = misc_utils.get_tensor_shape(logit)
    # gt_shape = get_tensor_shape(onehot_gt)

    # check sanity
    if config.num_of_classes == 1:
        raise ValueError("Number of classes is at least 2, background class must be included.")
    # if logit_shape != gt_shape:
    #     raise ValueError("Mismatching shapes between logits and ground_truths")
    if logit_shape[-1] != config.num_of_classes:
        raise ValueError("Something wrong. Check the get_shape of logit, ground_truth, and config.num_of_classes")

    # for calculating f1 loss with logit
    tp_logit = tf.reduce_sum(logit * onehot_gt, [0, 1, 2])
    fp_logit = tf.reduce_sum((1 - logit) * (1 - onehot_gt), [0, 1, 2])
    fn_logit = tf.reduce_sum(onehot_gt * (1 - logit), [0, 1, 2])
    precision_logit = tp_logit / (tp_logit + fp_logit + 1e-07)
    recall_logit = tp_logit / (tp_logit + fn_logit + 1e-07)
    f1 = 2 * precision_logit * recall_logit / (precision_logit + recall_logit + 1e-07)
    tf.add_to_collection("onehot_gt", onehot_gt)
    tf.add_to_collection("logit", logit)
    return 1 - tf.reduce_mean(f1)


def mf1_miou_loss(logit, ground_truth, config):
    # calculated f1 loss
    if ground_truth is None:
        raise ValueError("ground_truth cannot be None")
    onehot_gt = tf.one_hot(tf.cast(tf.squeeze(ground_truth, 3), tf.int32), config.num_of_classes)

    logit_shape = misc_utils.get_tensor_shape(logit)
    # gt_shape = get_tensor_shape(onehot_gt)

    # check sanity
    if config.num_of_classes == 1:
        raise ValueError("Number of classes is at least 2, background class must be included.")
    # if logit_shape != gt_shape:
    #     raise ValueError("Mismatching shapes between logits and ground_truths")
    if logit_shape[-1] != config.num_of_classes:
        raise ValueError("Something wrong. Check the get_shape of logit, ground_truth, and config.num_of_classes")

    # for calculating iou loss with logit
    intersection_logit = logit * onehot_gt  # [batch, height, width, class]
    union_logit = logit + onehot_gt - intersection_logit  # [batch, height, width, class]
    iou_logit = tf.reduce_sum(intersection_logit, [0, 1, 2]) / tf.reduce_sum(union_logit, [0, 1, 2])  # class
    miou_logit = tf.reduce_mean(iou_logit)

    # for calculating iou loss with logit
    tp_logit = tf.reduce_sum(logit * onehot_gt, [0, 1, 2])
    fp_logit = tf.reduce_sum((1 - logit) * (1 - onehot_gt), [0, 1, 2])
    fn_logit = tf.reduce_sum(onehot_gt * (1 - logit), [0, 1, 2])
    precision_logit = tp_logit / (tp_logit + fp_logit + 1e-07)
    recall_logit = tp_logit / (tp_logit + fn_logit + 1e-07)
    _logit = 2 * precision_logit * recall_logit / (precision_logit + recall_logit + 1e-07)
    mf1_logit = tf.reduce_mean(_logit)
    tf.add_to_collection("onehot_gt", onehot_gt)
    tf.add_to_collection("logit", logit)
    c1 = config.loss_fn_names[1][0]
    c2 = config.loss_fn_names[1][1]
    return 1 - (c1 * miou_logit + c2 * mf1_logit)  # mf1_miou_loss


def miou_recall_loss(logit, ground_truth, config):
    # calculated f1 loss
    if ground_truth is None:
        raise ValueError("ground_truth cannot be None")
    onehot_gt = tf.one_hot(tf.cast(tf.squeeze(ground_truth, 3), tf.int32), config.num_of_classes)

    logit_shape = misc_utils.get_tensor_shape(logit)
    # gt_shape = get_tensor_shape(onehot_gt)

    # check sanity
    if config.num_of_classes == 1:
        raise ValueError("Number of classes is at least 2, background class must be included.")
    # if logit_shape != gt_shape:
    #     raise ValueError("Mismatching shapes between logits and ground_truths")
    if logit_shape[-1] != config.num_of_classes:
        raise ValueError("Something wrong. Check the get_shape of logit, ground_truth, and config.num_of_classes")

    # for calculating iou loss with logit
    intersection_logit = logit * onehot_gt  # [batch, height, width, class]
    union_logit = logit + onehot_gt - intersection_logit  # [batch, height, width, class]
    iou_logit = tf.reduce_sum(intersection_logit, [0, 1, 2]) / tf.reduce_sum(union_logit, [0, 1, 2])  # class
    miou_logit = tf.reduce_mean(iou_logit)

    # for calculating iou loss with logit
    tp_logit = tf.reduce_sum(logit * onehot_gt, [0, 1, 2])
    fn_logit = tf.reduce_sum(onehot_gt * (1 - logit), [0, 1, 2])
    recall_logit = tp_logit / (tp_logit + fn_logit + 1e-07)
    mrecall_logit = tf.reduce_mean(recall_logit)
    c1 = config.loss_fn_names[1][0]
    c2 = config.loss_fn_names[1][1]
    tf.add_to_collection("onehot_gt", onehot_gt)
    tf.add_to_collection("logit", logit)
    return 1 - (c1 * miou_logit + c2 * mrecall_logit)  # mf1_miou_loss


def xntropy_loss(logit, ground_truth, config):
    if ground_truth is None:
        raise ValueError("No label is given")
    if config.mixup_proportion == 0.0:
        ground_truth = tf.reshape(ground_truth, shape=[-1])
        one_hot_labels = tf.cast(tf.one_hot(ground_truth, config.num_classes, on_value=1.0, off_value=0.0), logit.dtype)
    else:
        one_hot_labels = ground_truth

    if misc_utils.get_tensor_shape(logit) != misc_utils.get_tensor_shape(one_hot_labels):
        logit = tf.squeeze(logit)
        if misc_utils.get_tensor_shape(logit) != misc_utils.get_tensor_shape(one_hot_labels):
            logit = tf.reshape(logit, shape=[-1, config.num_classes])
            if misc_utils.get_tensor_shape(logit) != misc_utils.get_tensor_shape(one_hot_labels):
                raise ValueError("unexpted logit get_shape")
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logit))


def l1_loss(logit, ground_truth, *args):
    if ground_truth is None:
        raise ValueError("No label is given")
    if misc_utils.get_tensor_shape(logit) != misc_utils.get_tensor_shape(ground_truth):
        raise ValueError("Mismatching shapes between logits and ground_truths")
    return tf.reduce_mean(tf.abs(logit - ground_truth))


def l1_loss_with_box_kernel(logit, ground_truth, size):
    if ground_truth is None:
        raise ValueError("No label is given")
    if misc_utils.get_tensor_shape(logit) != misc_utils.get_tensor_shape(ground_truth):
        raise ValueError("Mismatching shapes between logits and ground_truths")
    window = tf.ones([size, size, 1, 1])
    window = window / (tf.reduce_sum(window))
    logit = tf.nn.conv2d(logit, window, [1, 1, 1, 1], "VALID")
    ground_truth = tf.nn.conv2d(ground_truth, window, [1, 1, 1, 1], "VALID")
    return tf.reduce_mean(tf.abs(logit - ground_truth))


def l1_loss_with_gaussian_kernel(logit, ground_truth, size, std):
    if ground_truth is None:
        raise ValueError("No label is given")
    if misc_utils.get_tensor_shape(logit) != misc_utils.get_tensor_shape(ground_truth):
        raise ValueError("Mismatching shapes between logits and ground_truths")
    window = misc_utils.gen_gaussian_kernel_by_sigma(size, std)
    logit = tf.nn.conv2d(logit, window, [1, 1, 1, 1], "VALID")
    ground_truth = tf.nn.conv2d(ground_truth, window, [1, 1, 1, 1], "VALID")
    return tf.reduce_mean(tf.abs(logit - ground_truth))


def l2_loss(logit, ground_truth, size=None, gaussian_kernel=False):
    # mean squared error
    if ground_truth is None:
        raise ValueError("No label is given")
    if misc_utils.get_tensor_shape(logit) != misc_utils.get_tensor_shape(ground_truth):
        raise ValueError("Mismatching shapes between logits and ground_truths")
    if size:  # no tf_kernel is applied if size=None
        if gaussian_kernel:
            window = misc_utils.gen_gaussian_kernel_by_sigma(size)
        else:
            window = tf.ones([size, size, 1, 1])
            window = window / (tf.reduce_sum(window))
        logit = tf.nn.conv2d(logit, window, [1, 1, 1, 1], "VALID")
        ground_truth = tf.nn.conv2d(ground_truth, window, [1, 1, 1, 1], "VALID")
    return tf.reduce_mean((logit - ground_truth) ** 2)


def rectified_mse_loss(logit, ground_truth):
    # todo:This is mostly likely trash
    if ground_truth is None:
        raise ValueError("No label is given")
    logit_shape = misc_utils.get_tensor_shape(logit)
    gt_shape = misc_utils.get_tensor_shape(ground_truth)
    if logit_shape != gt_shape:
        raise ValueError("Mismatching shapes between logits and ground_truths")
    with tf.name_scope("l2_loss"):
        rec_mse_zero = tf.where(tf.less(logit, 0.0), logit - ground_truth, tf.zeros_like(logit))
        rec_mse_one = tf.where(tf.greater(logit, 1.0), logit - ground_truth, tf.zeros_like(logit))
        rec_mse = rec_mse_zero + rec_mse_one
        rec_mse = tf.where(tf.equal(rec_mse, 0.0), (logit - ground_truth) ** 2, rec_mse)
        return tf.reduce_mean(rec_mse)


def ssim_components(img1, img2, kernel):
    # this seems to be wrong implementation of SSIM
    # sigma1_sq, sigma2_sq, and sigma12 use conv2d but mu1 and mu2 use depthwise_conv2d
    k1 = 0.01
    k2 = 0.03
    L = tf.reduce_max([img1, img2])  # 1  # depth of image (255 in case the image has a differnt scale)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    channel = misc_utils.get_tensor_shape(img1)[-1]
    kernel = tf.concat([kernel] * channel, 2)

    mu1 = tf.nn.depthwise_conv2d(img1, kernel, [1, 1, 1, 1], "VALID")
    mu2 = tf.nn.depthwise_conv2d(img2, kernel, [1, 1, 1, 1], "VALID")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.depthwise_conv2d(img1 * img1, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_sq
    sigma2_sq = tf.nn.depthwise_conv2d(img2 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu2_sq
    sigma12 = tf.nn.depthwise_conv2d(img1 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_mu2

    # sigma1_sq = tf.nn.conv2d(img1 * img1, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_sq
    # sigma2_sq = tf.nn.conv2d(img2 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu2_sq
    # sigma12 = tf.nn.conv2d(img1 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_mu2

    l_term = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    cs_term = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    lcs_term = l_term * cs_term
    return l_term, cs_term, lcs_term


def ssim_l_term_loss_with_box_kernel(img1, img2, size=11):
    l_term, _, _ = ssim_components(img1, img2, misc_utils.gen_box_kernel(size))
    return 1 - tf.reduce_mean(l_term)


def ssim_cs_term_loss_with_box_kernel(img1, img2, size=11):
    _, cs_term, _ = ssim_components(img1, img2, misc_utils.gen_box_kernel(size))
    return 1 - tf.reduce_mean(cs_term)


def ssim_lcs_term_loss_with_box_kernel(img1, img2, size=11):
    # same as ssim loss, but avg tf_kernel
    _, _, lcs_term = ssim_components(img1, img2, misc_utils.gen_box_kernel(size))
    return 1 - tf.reduce_mean(lcs_term)


def ssim_lcs_term_l1_loss_with_box_kernel(img1, img2, size=11, alpha=0.8):
    _, _, lcs_term = ssim_components(img1, img2, misc_utils.gen_box_kernel(size))
    lcs = 1 - tf.reduce_mean(lcs_term)
    l1 = l1_loss_with_box_kernel(img1, img2, misc_utils.gen_box_kernel(size))
    return alpha * lcs + (1 - alpha) * l1


def ssim_l_term_loss_with_gaussian(img1, img2, size=11):
    # original paper used the std of 1.5
    kernel = misc_utils.gen_gaussian_kernel_by_sigma(size, 1.5)
    l_term, _, _ = ssim_components(img1, img2, kernel)
    return 1 - tf.reduce_mean(l_term)


def ssim_cs_term_loss_with_gaussian(img1, img2, size=11):
    # original paper used the std of 1.5
    kernel = misc_utils.gen_gaussian_kernel_by_sigma(size, 1.5)
    _, cs_term, _ = ssim_components(img1, img2, kernel)
    return 1 - tf.reduce_mean(cs_term)


def ssim_lcs_term_loss_with_gaussian(img1, img2, size=11):
    # exactly same as ssim loss
    # original paper used the std of 1.5
    kernel = misc_utils.gen_gaussian_kernel_by_sigma(size, 1.5)
    _, _, lcs_term = ssim_components(img1, img2, kernel)
    return 1 - tf.reduce_mean(lcs_term)


def ms_ssim_cs_term_loss_with_random_gaussian(img1, img2, num_scales):
    random_kernel_size = misc_utils.get_random_gaussian_kernel_size(img1)
    random_scales = misc_utils.get_random_sigmas(random_kernel_size, num_scales)
    cs_terms = 1
    for scale in tf.unstack(random_scales):
        kernel = misc_utils.gen_gaussian_kernel_by_sigma(random_kernel_size, scale)
        _, cs_term, _ = ssim_components(img1, img2, kernel)
        cs_terms *= cs_term
    return 1 - tf.reduce_mean(cs_terms)


def ms_ssim_lcs_term_loss_with_random_gaussian(img1, img2, num_scales):
    random_kernel_size = misc_utils.get_random_gaussian_kernel_size(img1)
    random_scales = misc_utils.get_random_sigmas(random_kernel_size, num_scales)
    cs_terms = 1
    for scale in tf.unstack(random_scales):
        kernel = misc_utils.gen_gaussian_kernel_by_sigma(random_kernel_size, scale)
        l_term, cs_term, _ = ssim_components(img1, img2, kernel)
        cs_terms *= cs_term
    lcs_term = l_term * cs_terms
    return 1 - tf.reduce_mean(lcs_term)


def ms_ssim_lcs_term_loss_with_random_gaussian_pseudo_huber(img1, img2, num_scales):
    random_kernel_size = misc_utils.get_random_gaussian_kernel_size(img1)
    random_scales = misc_utils.get_random_sigmas(random_kernel_size, num_scales)
    cs_terms = 1
    for scale in tf.unstack(random_scales):
        kernel = misc_utils.gen_gaussian_kernel_by_sigma(random_kernel_size, scale)
        l_term, cs_term, _ = ssim_components(img1, img2, kernel)
        cs_terms *= cs_term
    lcs_term = l_term * cs_terms
    huber_term = pseudo_huber_loss(img2, img1)
    combined_loss = (1 - tf.reduce_mean(lcs_term)) * 0.7 + huber_term * 0.3
    return combined_loss


####
# todo: experimental

def ridge_components(img, kernel):
    img_kernel_filtered = tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID")
    dx_kernel = tf.constant([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]], tf.float32)
    dx_kernel = dx_kernel[:, :, tf.newaxis, tf.newaxis]
    dy_kernel = tf.constant([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]], tf.float32)
    dy_kernel = dy_kernel[:, :, tf.newaxis, tf.newaxis]
    dy = tf.nn.conv2d(img_kernel_filtered, dy_kernel, [1, 1, 1, 1], padding="SAME")
    dx = tf.nn.conv2d(img_kernel_filtered, dx_kernel, [1, 1, 1, 1], padding="SAME")
    dydy = tf.nn.conv2d(dy, dy_kernel, [1, 1, 1, 1], padding="SAME")
    dydx = tf.nn.conv2d(dy, dx_kernel, [1, 1, 1, 1], padding="SAME")
    dxdx = tf.nn.conv2d(dx, dx_kernel, [1, 1, 1, 1], padding="SAME")
    eq1 = (dydy + dxdx) / 2
    eq2 = tf.sqrt(4 * dydx ** 2 + (dydy - dxdx) ** 2) / 2
    ridge_maxima = eq1 + eq2
    ridge_minima = eq1 - eq2
    return ridge_maxima, ridge_minima


def ridge_similarity(img1, img2, kernel):
    img1_ridge, _ = ridge_components(img1, kernel)
    img2_ridge, _ = ridge_components(img2, kernel)
    # img1_ridge = tf.nn.relu(img1_ridge)
    # img2_ridge = tf.nn.relu(img2_ridge)
    max = tf.reduce_max([img1_ridge, img2_ridge])
    c2 = 0.001
    coefficient = (2 * img1_ridge * img2_ridge + c2) / (img1_ridge ** 2 + img2_ridge ** 2 + c2)
    return coefficient


def pseudo_huber_loss(label, pred, *args):
    # delta = 2.0
    delta = 1.0
    return tf.reduce_mean(tf.multiply(tf.square(delta), tf.sqrt(1. + tf.square((label - pred) / delta)) - 1.))


def pseudo_huber_loss_with_kernel(label, pred, kernel):
    delta = 2.0
    channel = misc_utils.get_tensor_shape(pred)[-1]
    kernel = tf.concat([kernel] * channel, 2)

    label = tf.nn.depthwise_conv2d(label, kernel, [1, 1, 1, 1], "VALID")
    pred = tf.nn.depthwise_conv2d(pred, kernel, [1, 1, 1, 1], "VALID")
    return tf.reduce_mean(tf.multiply(tf.square(delta), tf.sqrt(1. + tf.square((label - pred) / delta)) - 1.))


def ridge_pseudo_huber_loss(label, pred, *args):
    delta = 0.25
    label = misc_utils.detect_ridge_channel_wise(label)
    pred = misc_utils.detect_ridge_channel_wise(pred)
    return tf.reduce_mean(tf.multiply(tf.square(delta), tf.sqrt(1. + tf.square((label - pred) / delta)) - 1.))


def ridge_l1_loss(label, pred, *args):
    label = misc_utils.detect_ridge_channel_wise(label)
    pred = misc_utils.detect_ridge_channel_wise(pred)
    tf.add_to_collection("ridge_pred", pred)
    tf.add_to_collection("ridge_gt", label)
    return l1_loss(pred, label)


def ridge_loss_maxima_with_random_gaussian(img1, img2, kernel):
    ridge_similarity_coefficient, _ = ridge_similarity(img1, img2, kernel)
    return 1 - ridge_similarity_coefficient


def ms_ssim_cs_term_and_ridge_term_with_random_gaussian(img1, img2, num_scales):
    random_kernel_size = misc_utils.get_random_gaussian_kernel_size(img1)
    random_scales = misc_utils.get_random_sigmas(random_kernel_size, num_scales)
    cs_and_r_terms = 1
    for scale in tf.unstack(random_scales):
        kernel = misc_utils.gen_gaussian_kernel_by_sigma(random_kernel_size, scale)
        _, cs_term, _ = ssim_components(img1, img2, kernel)
        ridge_similarity_coefficient = ridge_similarity(img1, img2, kernel)
        cs_and_r_terms *= cs_term * ridge_similarity_coefficient
    return 1 - tf.reduce_mean(cs_and_r_terms)


def ssim_ridge_components(img1, img2, kernel):
    k1 = 0.01
    k2 = 0.03
    L = tf.reduce_max([img1, img2])  # 1  # depth of image (255 in case the image has a differnt scale)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    channel = misc_utils.get_tensor_shape(img1)[-1]
    kernel = tf.concat([kernel] * channel, 2)

    mu1 = tf.nn.depthwise_conv2d(img1, kernel, [1, 1, 1, 1], "VALID")
    mu2 = tf.nn.depthwise_conv2d(img2, kernel, [1, 1, 1, 1], "VALID")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    # negative values are supposed to be zero mathematically,
    # and tf.nn.relu is applied to set negative values to zeros
    sigma1_sq = tf.nn.conv2d(img1 * img1, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_mu2

    l_term = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    cs_term = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    r_term = ridge_similarity(img1, img2, kernel)
    return l_term, cs_term, r_term


def ms_ssim_lcs_term_loss_and_ridge_loss_with_random_gaussian(img1, img2, num_scales):
    random_kernel_size = misc_utils.get_random_gaussian_kernel_size(img1)
    random_scales = misc_utils.get_random_sigmas(random_kernel_size, num_scales)
    r_terms = []
    cs_term = 1
    for scale in tf.unstack(random_scales):
        kernel = misc_utils.gen_gaussian_kernel_by_sigma(random_kernel_size, scale)
        l_term, cs_term, r_term = ssim_ridge_components(img1, img2, kernel)
        cs_term *= cs_term * r_term
        r_terms.append(tf.reduce_mean(r_term))
    lcs_term = l_term * cs_term
    tf.add_to_collection("r_terms", r_terms)
    return 1 - tf.reduce_mean(lcs_term)


def ms_ssim_lcs_term_loss_and_l1_loss_with_random_gaussian(img1, img2, num_scales):
    random_kernel_size = misc_utils.get_random_gaussian_kernel_size(img1)
    random_scales = misc_utils.get_random_sigmas(random_kernel_size, num_scales)
    cs_terms = 1
    for scale in tf.unstack(random_scales):
        kernel = misc_utils.gen_gaussian_kernel_by_sigma(random_kernel_size, scale)
        l_term, cs_term, _ = ssim_components(img1, img2, kernel)
        cs_terms *= cs_term
    l1 = l1_loss(img1, img2)
    lcs_term = tf.reduce_mean(l_term * cs_terms)
    loss = lcs_term * 0.9 + l1 * 0.1
    return 1 - loss


def ssim_sobel(img1, img2, kernel):
    k1 = 0.01
    k2 = 0.03
    L = tf.reduce_max([img1, img2])  # 1  # depth of image (255 in case the image has a differnt scale)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    channel = misc_utils.get_tensor_shape(img1)[-1]
    kernel = tf.concat([kernel] * channel, 2)

    mu1 = tf.nn.depthwise_conv2d(img1, kernel, [1, 1, 1, 1], "VALID")
    mu2 = tf.nn.depthwise_conv2d(img2, kernel, [1, 1, 1, 1], "VALID")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_mu2

    l_term = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    cs_term = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    lcs_term = l_term * cs_term

    sobel1 = misc_utils.sobel(mu1)
    sobel2 = misc_utils.sobel(mu2)

    sobel_term = (2 * sobel1 * sobel2 + c2) / (sobel1 ** 2 + sobel2 ** 2 + c2)

    return l_term, cs_term, sobel_term


def ssim_sobel2(img1, img2, kernel, kernel_size, reference_kernel_size):
    k1 = 0.01
    k2 = 0.03
    L = tf.reduce_max([img1, img2])  # 1  # depth of image (255 in case the image has a differnt scale)
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    channel = misc_utils.get_tensor_shape(img1)[-1]
    kernel = tf.concat([kernel] * channel, 2)

    clip_size = (reference_kernel_size - kernel_size) / 2

    def clip_image(image):
        return image[:, clip_size:-clip_size, clip_size:-clip_size, :]

    img1 = tf.cond(tf.equal(clip_size, 0), lambda: img1, lambda: clip_image(img1))
    img2 = tf.cond(tf.equal(clip_size, 0), lambda: img2, lambda: clip_image(img2))

    mu1 = tf.nn.depthwise_conv2d(img1, kernel, [1, 1, 1, 1], "VALID")
    mu2 = tf.nn.depthwise_conv2d(img2, kernel, [1, 1, 1, 1], "VALID")
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, kernel, strides=[1, 1, 1, 1], padding="VALID") - mu1_mu2

    l_term = (2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)
    cs_term = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)

    sobel1 = misc_utils.sobel(mu1)
    sobel2 = misc_utils.sobel(mu2)

    sobel_term = (2 * sobel1 * sobel2 + c2) / (sobel1 ** 2 + sobel2 ** 2 + c2)

    return l_term, cs_term, sobel_term


# def rms_ssim_sobel_loss(img1, img2, num_scales):
#     # VERY IMPORTANT NOTE:
#     # this function is DIFFERENT from other rms_ssim loss functions within this file
#     # this function generate random kernel sizes at each scales while others uses fixed kernel size at each iteration.
#     random_kernel_sizes = tf.unstack(misc_utils.get_random_kernel_sizes(img1, num_scales))
#     cs_terms = 1
#     for i, kernel_size in enumerate(random_kernel_sizes):
#         random_sigma = misc_utils.get_random_sigmas(kernel_size, 1)
#         kernel = misc_utils.gaussian_kernel_2d(kernel_size, random_sigma)
#         kernel = misc_utils.remap_kernel(kernel, kernel_size, random_kernel_sizes[-1])
#         if i == 0:
#             _, cs_term, s_term = ssim_sobel(img1, img2, kernel)
#         elif i == len(tf.unstack(random_sigma)):
#             l_term, cs_term, _ = ssim_sobel(img1, img2, kernel)
#         else:
#             _, cs_term, _ = ssim_sobel(img1, img2, kernel)
#         cs_terms *= cs_term
#     rms_ssim_sobel = l_term * cs_terms * s_term
#     return 1 - tf.reduce_mean(rms_ssim_sobel)


def rms_ssim_sobel_loss(img1, img2, num_scales):
    # VERY IMPORTANT NOTE:
    # this function is DIFFERENT from other rms_ssim loss functions within this file
    # this function generate random kernel sizes at each scales while others uses fixed kernel size at each iteration.
    # Result: not good
    random_kernel_sizes = tf.unstack(misc_utils.get_random_kernel_sizes(img1, num_scales))
    cs_terms = 1
    for i, kernel_size in enumerate(random_kernel_sizes):
        random_sigma = misc_utils.get_random_sigmas(kernel_size, 1)
        kernel = misc_utils.gaussian_kernel_2d(kernel_size, random_sigma)
        kernel = misc_utils.remap_kernel(kernel, kernel_size, random_kernel_sizes[-1])
        l_term, cs_term, _ = ssim_sobel(img1, img2, kernel)
        cs_terms *= cs_term
    rms_ssim_sobel = l_term * cs_terms
    return 1 - tf.reduce_mean(rms_ssim_sobel)


def sobel_loss(img1, img2, dummy):
    kernel_size = 7
    random_sigma = misc_utils.get_random_sigmas(kernel_size, 1)
    kernel = misc_utils.gaussian_kernel_2d(kernel_size, random_sigma)
    kernel = tf.concat([kernel] * 3, 2)
    denoised_img1 = tf.nn.depthwise_conv2d(img1, kernel, [1, 1, 1, 1], "VALID")
    denoised_img2 = tf.nn.depthwise_conv2d(img2, kernel, [1, 1, 1, 1], "VALID")

    sobel1 = misc_utils.sobel(img1)
    sobel2 = misc_utils.sobel(img2)
    tf.add_to_collection("sobel_img", sobel1)
    tf.add_to_collection("sobel_gt", sobel2)
    return tf.reduce_mean((sobel1 - sobel2) ** 2)
