import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2

slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS


def get_metric_table(cmatrix):
    TP = tf.diag_part(cmatrix)
    FP = tf.reduce_sum(cmatrix, 0) - TP
    FN = tf.reduce_sum(cmatrix, 1) - TP
    table = tf.concat([TP[:, None], FP[:, None], FN[:, None]], 1)
    Precision = table[:, 0] / (table[:, 0] + table[:, 1])
    Recall = table[:, 0] / (table[:, 0] + table[:, 2])
    F1 = 2 * Precision * Recall / (Precision + Recall)
    iou = TP / (tf.reduce_sum(table))

    tf.add_to_collection("custom_metrics", tf.identity(Precision[-1], name="precision"))
    tf.add_to_collection("custom_metrics", tf.identity(Recall[-1], name="recall"))
    tf.add_to_collection("custom_metrics", tf.identity(F1[-1], name="F1"))
    tf.add_to_collection("custom_metrics", tf.identity(iou[-1], name="iou"))
    tf.add_to_collection("custom_metrics", tf.identity(TP[-1], name="TP"))
    tf.add_to_collection("custom_metrics", tf.identity(FP[-1], name="FP"))


def iou_loss(logits,
             labels,
             upsample_logits=True):
    if labels is None:
        raise ValueError("No label is given")

    if upsample_logits:
        # Label is not downsampled, and instead we upsample logits.
        logits = tf.image.resize_bilinear(
            logits, tf.shape(labels)[1:3], align_corners=True)
        scaled_labels = labels
    else:
        # Label is downsampled to the same size as logits.
        scaled_labels = tf.image.resize_nearest_neighbor(
            labels, tf.shape(logits)[1:3], align_corners=True)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    logits = tf.reshape(logits, [-1])
    scaled_labels = tf.cast(scaled_labels, tf.float32)
    # for calculating loss
    intersection = tf.reduce_sum(tf.multiply(logits, scaled_labels))
    union = tf.reduce_sum(tf.subtract(tf.add(logits, scaled_labels), tf.multiply(logits, scaled_labels)))
    loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(intersection, union))

    # for monitoring iou
    pred = tf.cast(tf.greater(logits, 0.5), dtype=tf.float32)
    inter = tf.reduce_sum(tf.multiply(pred, scaled_labels))
    uni = tf.cast(tf.reduce_sum(tf.subtract(tf.add(pred, scaled_labels), tf.multiply(pred, scaled_labels))),
                  dtype=tf.float32)
    iou = tf.div(inter, uni)
    tf.add_to_collection("losses", tf.identity(loss, name="iou_loss"))
    tf.add_to_collection("custom_metrics", tf.identity(iou, name="batch_mIoU"))


def xntropy_loss(logits,
                 labels,
                 num_classes):
    if labels is None:
        raise ValueError("No label for softmax cross entropy loss.")

    labels = tf.reshape(labels, shape=[-1])
    one_hot_labels = tf.one_hot(labels, num_classes, on_value=1.0, off_value=0.0)

    tf.losses.softmax_cross_entropy(
        one_hot_labels,
        tf.reshape(logits, shape=[-1, num_classes]),
        scope="xntropy")

    if FLAGS.task == "classification":
        tf_pred = tf.argmax(logits, 1)
        tf_top1 = tf.math.in_top_k(logits, labels, 1, name="top1")
        tf_top5 = tf.math.in_top_k(logits, labels, 5, name="top5")
        tf.add_to_collection("get_logit", tf.identity(logits, name="get_logit"))
        tf.add_to_collection("prediction", tf.identity(tf_pred, name="prediction"))
        tf.add_to_collection("top1_tp", tf.identity(tf_top1, name="top1_tp"))
        tf.add_to_collection("top5_tp", tf.identity(tf_top5, name="top5_tp"))
