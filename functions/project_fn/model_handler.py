from functions.project_fn.utils import get_shape, list_getter
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc, imwrite
from functions.project_fn.module import Module
from math import pi, isnan, isinf
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import os
import time


class TrainHandler:
    """
    a parent class of ModelHandler
    """

    def _build_summary_op(self):
        for index, grad in enumerate(self.grads_and_vars):
            tf.summary.histogram("{}-grad".format(self.grads_and_vars[index][1].name), self.grads_and_vars[index][0])
            tf.summary.histogram(self.grads_and_vars[index][1].name, self.grads_and_vars[index][1])
        tf.summary.scalar("mIoU loss", self.loss)
        tf.summary.scalar("learning rate", self.lr)
        tf.summary.scalar("batch size", self.batch_size)

    def _miou_loss(self):
        # calculated bache mean intersection over union loss
        if self.dtype == tf.float16:
            logit = tf.cast(self.logit, tf.float32)
        else:
            logit = self.logit
        prob_map = tf.nn.softmax(logit)
        onehot_gt = tf.one_hot(tf.cast(tf.squeeze(self.gt, 3), tf.uint8), self.num_classes)

        # calculate iou loss
        intersection_logit = prob_map * onehot_gt  # [batch, height, width, class]
        union_logit = prob_map + onehot_gt - intersection_logit  # [batch, height, width, class]
        iou_logit = tf.reduce_sum(intersection_logit, [0, 1, 2]) / tf.reduce_sum(union_logit, [0, 1, 2])  # class
        miou_logit = tf.reduce_mean(iou_logit)
        self.loss = 1.0 - tf.reduce_mean(miou_logit)

    def _get_learning_rate(self):
        global_step = tf.cast(self.global_step, tf.float64)
        const_0 = tf.constant(0.0, dtype=tf.float64)
        const_1 = tf.constant(1, dtype=tf.float64)
        const_2 = tf.constant(2, dtype=tf.float64)
        slow_start_step_size = tf.constant(self.slow_start_step_size, tf.float64)
        cycle_step_size = tf.constant(self.cycle_step_size, tf.float64)

        max_lr = tf.constant(self.max_lr, tf.float64)
        min_lr = tf.constant(self.min_lr, tf.float64)
        max_lr_decay_step = tf.cond(tf.less_equal(global_step, slow_start_step_size),
                                    lambda: const_0,
                                    lambda: tf.floor(const_1 + (global_step - slow_start_step_size) / cycle_step_size))

        max_lr_decay = tf.constant(self.max_lr_decay, tf.float64)
        max_lr = max_lr * (max_lr_decay ** (max_lr_decay_step - const_1))
        cos_inner = (tf.constant(pi, tf.float64) * tf.floormod(global_step - slow_start_step_size, cycle_step_size)) / cycle_step_size

        self.lr = tf.cast(tf.cond(tf.less_equal(global_step, slow_start_step_size),
                                  lambda: self.min_lr + (self.max_lr - self.min_lr) / slow_start_step_size * global_step,
                                  lambda: (max_lr - min_lr) / const_2 * (tf.cos(cos_inner) + const_1) + min_lr), tf.float32)

    def _build_train_op(self, optimizer):
        self.grads_and_vars = optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables())
        none_grad_vars = []
        for grad, var in self.grads_and_vars:
            if grad is None:
                none_grad_vars.append(var)
        if none_grad_vars:
            for var in none_grad_vars:
                print(var.name)
            raise ValueError('The above variables have no gradient')
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

    def _train_step(self, graph, sess, saver):
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.ckpt_dir, graph=graph)

        print('Start training...')
        global_step = sess.run(self.global_step)

        should_continue = True if global_step <= self.max_step else False
        while should_continue:
            start_time = time.time()
            _, batch_loss, global_step, lr = sess.run([self.train_op, self.loss, self.global_step, self.lr])
            elapsed = time.time() - start_time

            # check if loss value is nan or inf
            should_terminate = isnan(batch_loss) or isinf(batch_loss)

            is_at_lr_transition = True if global_step > self.cycle_step_size + self.slow_start_step_size and (
                    global_step + self.slow_start_step_size) % self.cycle_step_size in [1, 0] else False

            if not global_step % self.log_print_interval:
                print('step=%d(%.3f sec/step), total loss=%.3f, lr=%.9f' % (global_step, elapsed, batch_loss, lr))

            if not global_step % self.ckpt_save_interval or is_at_lr_transition:
                save_path = self.ckpt_dir + "/" + "model_step"
                saver.save(sess, save_path, global_step=global_step, write_meta_graph=False)
                print("model is saved")
            #
            if not global_step % self.summary_save_interval or is_at_lr_transition:
                summary_writer.add_summary(sess.run(summary_op), global_step)
                print("summary is saved")
            #
            if should_terminate:
                raise ValueError('Model diverged with loss = %s' % batch_loss)

            should_continue = True if global_step <= self.max_step else False

    def _start_train(self, hvd, sess):
        graph = tf.get_default_graph()
        saver = tf.train.Saver(max_to_keep=5000)
        with graph.as_default() as graph:
            global_init_fn = tf.global_variables_initializer()
            local_init_fn = tf.local_variables_initializer()
            init_fn = tf.group(global_init_fn, local_init_fn)
            all_ckpt_list = [_.split(".index")[0] for _ in list_getter(self.ckpt_dir, 'index')]
            sess.run(init_fn)
            if all_ckpt_list:  # assumed the current model is intended to continue training if latest checkpoint exists
                print('Training will be continued from the last checkpoint...')
                saver.restore(sess, all_ckpt_list[-1])
                print('The last checkpoint is loaded!')
            else:
                print('Training will be started from scratch...')
            sess.run(hvd.broadcast_global_variables(0))
            self._train_step(graph, sess, saver)

    def _train_handler(self, hvd, sess):
        self._miou_loss()
        self.global_step = tf.train.get_or_create_global_step()
        self._get_learning_rate()
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)

        if self.dtype == tf.float16:
            loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(128, 100)
            # Wraps the original optimizer in a LossScaleOptimizer.
            optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
            compression = hvd.Compression.fp16
        elif self.dtype == tf.float32:
            compression = hvd.Compression.none
        else:
            raise ValueError('unexpected dtype')
        optimizer = hvd.DistributedOptimizer(optimizer, compression=compression)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self._build_train_op(optimizer)
        self._build_summary_op()
        self._start_train(hvd, sess)


class EvalHandler:
    """
    a parent class of ModelHandler
    """

    def _init_log(self):
        with open(os.path.join(self.eval_log_dir, 'metric_overall.csv'), 'a+') as writer:
            writer.seek(0)  # python 3, this line must be included. it's a python bug.
            log = writer.readlines()
            if not log:
                if self.num_classes <= 2:
                    writer.write('ckpt_id, precision, recall, f1, miou\n')
                else:
                    writer.write('ckpt_id, miou\n')
            else:
                log = [entry.strip() for entry in log]
        self.log = log

    def _get_ckpt_in_range(self):
        all_ckpt_list = [_.split(".index")[0] for _ in list_getter(self.ckpt_dir, 'index')]
        ckpt_pattern = './model/checkpoints/model_step-%d'
        if self.ckpt_start == 'beginning':
            start_idx = 0
        else:
            start_idx = all_ckpt_list.index(ckpt_pattern % self.ckpt_start)

        if self.ckpt_end == 'end':
            end_idx = None
        else:
            end_idx = all_ckpt_list.index(ckpt_pattern % self.ckpt_end) + 1
        return all_ckpt_list[start_idx:end_idx:self.ckpt_step]

    def _calculate_segmentation_metric(self):
        tp = np.diag(self.cumulative_cmatrix)
        fp = np.sum(self.cumulative_cmatrix, axis=0) - tp
        fn = np.sum(self.cumulative_cmatrix, axis=1) - tp
        precision = tp / (tp + fp)  # precision of each class. [batch, class]
        recall = tp / (tp + fn)  # recall of each class. [batch, class]
        f1 = 2 * precision * recall / (precision + recall)
        iou = tp / (tp + fp + fn)  # iou of each class. [batch, class]
        miou = iou.mean()  # miou
        if iou.shape[0] <= 2:
            self.metrics = [precision[1], recall[1], f1[1], miou]
        else:
            self.metrics = [miou]

    def _write_eval_log(self, ckpt_id):
        with open(os.path.join(self.eval_log_dir, 'metric_overall.csv'), 'a+') as writer:
            writer.write('%s, ' % ckpt_id)
            writer.write(', '.join([str(value) for value in self.metrics]) + '\n')

    def _eval(self, sess, ckpt_id):
        while True:
            try:
                self.cumulative_cmatrix += sess.run(self.confusion_matrix)
            except tf.errors.OutOfRangeError:
                self._calculate_segmentation_metric()
                self._write_eval_log(ckpt_id)
                break

    def _eval_handler(self, sess):
        restorer = tf.train.Saver()
        pred = tf.expand_dims(tf.argmax(self.logit, 3), 3)
        self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.gt, [-1]),
                                                    tf.reshape(pred, [-1]),
                                                    self.num_classes,
                                                    dtype=tf.float32)
        for ckpt in self._get_ckpt_in_range():
            self._init_log()
            self.cumulative_cmatrix = np.zeros((self.num_classes, self.num_classes))
            ckpt_id = os.path.basename(ckpt)
            if ckpt_id in [row.split(',')[0] for row in self.log[1:]]:
                print('Log for the current ckpt (%s) already exsit. This ckpt is skipped' % ckpt_id)
            else:
                print('Current ckpt: %s' % ckpt)
                restorer.restore(sess, ckpt)
                sess.run(self.data_init)
                self._eval(sess, ckpt_id)


class VisHandler:
    """
    a parent class of ModelHandler
    """

    def _get_ckpt(self):
        all_ckpt_list = [_.split(".index")[0] for _ in list_getter(self.ckpt_dir, 'index')]
        ckpt_pattern = './model/checkpoints/model_step-%d'
        return all_ckpt_list[all_ckpt_list.index(ckpt_pattern % self.ckpt_id)]

    def _vis_with_image(self, sess):
        while True:
            try:
                img, pred, filename = sess.run([tf.squeeze(self.input_data),
                                                tf.squeeze(self.pred),
                                                tf.squeeze(self.filename)])
                basename = os.path.basename(filename.decode("utf-8"))
                mask = np.ones_like(pred) - pred
                color_label = np.stack([np.zeros_like(pred), np.zeros_like(pred), pred * 255], 2)
                superimposed = img[:, :, ::-1] * np.expand_dims(mask, 2) + color_label
                dst_name = self.vis_result_dir + "/" + basename
                imwrite(dst_name, superimposed)
            except tf.errors.OutOfRangeError:
                break

    def _vis_with_video(self, sess):
        # self.img_dir is actually video dir
        # I tested only avi and mp4 so far
        vid_list = list_getter(self.img_dir, ("avi", "mp4"))
        for vid_name in vid_list:
            vid = VideoCapture(vid_name)
            should_continue, frame = vid.read()
            basename = os.path.basename(vid_name)[:-4]
            dst_name = self.vis_result_dir + "/" + basename + ".mp4"

    def _vis_handler(self, sess):
        restorer = tf.train.Saver()
        self.pred = tf.squeeze(tf.argmax(self.logit, 3))
        restorer.restore(sess, self._get_ckpt())
        sess.run(self.data_init)
        if self.data_type == "image":
            self._vis_with_image(sess)
        elif self.data_type == "video":
            self._vis_with_video(sess)
        else:
            raise ValueError("Unexpected data_type")


class ModelHandler(Module, TrainHandler, EvalHandler, VisHandler):
    def __init__(self, data, config):
        self.config = config
        super(ModelHandler, self).__init__()
        self.dtype = tf.float16 if self.dtype == "fp16" else tf.float32
        self.input_data = data.input_data  #
        self.gt = data.gt  # this will be none in case phase=vis, data_type=video
        self.filename = data.filename
        self.data_init = data.data_init
        self._build_model()

    def __getattr__(self, item):
        try:
            return getattr(self.config, item)
        except AttributeError:
            raise AttributeError("'config' has no attribute '%s'" % item)

    @staticmethod
    def fp32_var_getter(getter,
                        name,
                        shape=None,
                        dtype=None,
                        initializer=None,
                        regularizer=None,
                        trainable=True,
                        *args, **kwargs):
        """Custom variable getter that forces trainable variables to be stored in
        float32 precision and then casts them to the training precision.
        """
        variable = getter(name,
                          shape,
                          dtype=tf.float32 if trainable else dtype,
                          initializer=initializer,
                          regularizer=regularizer,
                          trainable=trainable,
                          *args, **kwargs)
        if trainable and dtype != tf.float32:
            variable = tf.cast(variable, dtype)
        return variable

    def architecture_fn(self):
        normalized_input = (tf.cast(self.input_data, self.dtype) / 127.5 - 1) * 1.3
        with tf.device("/GPU:0"), tf.variable_scope("fp32_var", custom_getter=self.fp32_var_getter, use_resource=True, reuse=False):
            root = self.convolution(normalized_input, 5, 1, 16, "root")
            en1, fp_feature1 = self.squeezing_dense(root, [16, 32], [5, 3], [1, 2], "encoder1")
            net = self.shortcut(en1, root, 3, 2, get_shape(root)[-1] / 2, "shortcut_concat1")
            en2, _ = self.squeezing_dense(net, [16, 32, 48], [7, 5, 3], [1, 1, 2], "encoder2", True, 4)
            en3, fp_feature2 = self.squeezing_dense(en2, [16, 32, 48, 64], [9, 7, 5, 3], [1, 1, 1, 2], "encoder3", True, 4)
            net = self.shortcut(en3, en2, 3, 2, get_shape(en2)[-1] / 2, "shortcut_concat2")
            en4, _ = self.squeezing_dense(net, [16, 32, 48, 64, 80], [11, 9, 7, 5, 3], [1, 1, 1, 1, 2], "encoder4", True, 4)
            repeat = en4
            for j in range(4):
                repeat, _ = self.squeezing_dense(repeat, [16, 32, 48, 64, 80], [11, 9, 7, 5, 3], [1, 1, 1, 1, 1], "encoder%d" % (j + 5), True, 4)
            net = self.upscale(repeat, fp_feature2, 4, 4, 24, "upsample1")
            net = self.convolution(net, 3, 1, 24, "decode1")
            net = self.upscale(net, fp_feature1, 4, 4, 16, "upsample2")
            self.logit = self.get_logit(net, 3, 1)

    def _build_model(self):
        hvd.init()
        # Using the Winograd non-fused algorithms provides a small performance boost.
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        print("Deploying model to GPU:%d..." % self.physical_gpu_id)
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        session_config.allow_soft_placement = True
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())
        sess = tf.Session(config=session_config)
        self.architecture_fn()
        if self.phase == "train":
            self._train_handler(hvd, sess)
        elif self.phase == "eval":
            self._eval_handler(sess)
        elif self.phase == "vis":
            self._vis_handler(sess)
        else:
            raise ValueError("Unexpected phase:%s" % self.phase)
