from functions.project_fn.module import Module
from functions.project_fn.utils import get_tensor_shape as get_shape
from functions.project_fn.utils import get_all_ckpt_list, fp32_var_getter
from functions.project_fn.train_handler import TrainHandler
import horovod.tensorflow as hvd
import tensorflow as tf
import numpy as np
import os
import time


class ModelHandler(Module, TrainHandler):
    def __init__(self, data, config):
        self.config = config
        self.image = data.image
        self.input = (tf.cast(self.image, self.dtype) / 127.5 - 1) * 1.3
        self.gt = data.gt
        self.hvd = hvd
        if self.phase != "_train":
            self.data_init = data.init
        self._build()

    def __getattr__(self, item):
        try:
            return getattr(self.config, item)
        except AttributeError:
            raise AttributeError("'config' has no attribute '%s'" % item)

    def architecture_fn(self):
        root = self.convolution(self.input, 5, 1, 16, "root")
        en1, fp_feature1 = self.squeezing_dense(root, [16, 32], [5, 3], [1, 2], "encoder1")
        net = self.shortcut(en1, root, 3, 2, get_shape(root)[-1] / 2, "shortcut_concat1")
        en2, _ = self.squeezing_dense(net, [16, 32, 48], [7, 5, 3], [1, 1, 2], "encoder2", True, 4)
        en3, fp_feature2 = self.squeezing_dense(en2, [16, 32, 48, 64], [9, 7, 5, 3], [1, 1, 1, 2], "encoder3", True, 4)
        net = self.shortcut(en3, en2, 3, 2, get_shape(en2)[-1] / 2, "shortcut_concat2")
        en4, _ = self.squeezing_dense(net, [16, 32, 48, 64, 80], [11, 9, 7, 5, 3], [1, 1, 1, 1, 2], "encoder4", True, 4)
        repeat = en4
        for j in range(4):
            repeat, _ = self.squeezing_dense(repeat, [16, 32, 48, 64, 80], [11, 9, 7, 5, 3], [1, 1, 1, 1, 1], "encoder%d" % (j + 5), True, 4)
        net = self.transpose_conv(repeat, fp_feature2, 4, 4, 24, "upsample1")
        net = self.convolution(net, 3, 1, 24, "decode1")
        net = self.transpose_conv(net, fp_feature1, 4, 4, 16, "upsample2")
        self.logit = self.get_logit(net, 3, 1)

    def _build(self):
        self.hvd.init()
        # Using the Winograd non-fused algorithms provides a small performance boost.
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        print("Deploying model to GPU:%d..." % self.physical_gpu_id)
        with tf.device("/GPU:0"), tf.variable_scope("fp32_var", custom_getter=fp32_var_getter, use_resource=True, reuse=False):
            self.architecture_fn()
            if self.phase == "_train":
                self._train()
            elif self.phase == 'eval':
                self.pred = tf.expand_dims(tf.argmax(self.logit, 3), 3)
                self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.gt, [-1]),
                                                            tf.reshape(self.pred, [-1]),
                                                            self.num_classes,
                                                            dtype=tf.float32)
            elif self.phase == "vis":
                self.pred = tf.squeeze(tf.argmax(self.logits, 3))
            else:
                raise ValueError('Unexpected phase')

    def get_variables(scope):
        return [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if scope in var.name]

    def get_variables_to_restore(exclude_list):
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vars_to_exclude = set()

        for scope in exclude_list:
            var_to_exclude = get_variables(scope)
            if not var_to_exclude:
                raise ValueError("'%s' does not exist in the current graph" % scope)
            vars_to_exclude |= set(var_to_exclude)
        return [v for v in global_variables if v not in vars_to_exclude]

    def save_checkpoint_and_summary(saver, sess, summary_writer, summary_op, step, config):
        if config.lr_policy in ["slow_start", "fixed"]:
            if step % config.ckpt_save_interval == 0 or step >= config.max_step or step == 1:
                saver.save(sess, os.path.join(config.ckpt_dir, "model_step"), global_step=step, write_meta_graph=False)
                print("The checkpoint at step = %d is saved" % step)
        elif config.lr_policy == "cyclical":
            if step % config.ckpt_save_interval == 0 or step >= config.max_step or step == 1 or step % (config.cycle_step_size - 1) == 0:
                saver.save(sess, os.path.join(config.ckpt_dir, "model_step"), global_step=step, write_meta_graph=False)
                print("The checkpoint at step = %d is saved" % step)
        else:
            raise ValueError("unexpected lr_policy")

        if config.lr_policy in ["slow_start", "fixed"]:
            if step % config.summary_save_interval == 0 or step >= config.max_step or step == 1:
                summary_writer.add_summary(sess.run(summary_op), step)
                print("The summary at step = %d is saved" % step)
        elif config.lr_policy == "cyclical":
            if step % config.summary_save_interval == 0 or step >= config.max_step or step == 1 or step % (config.cycle_step_size - 1) == 0:
                summary_writer.add_summary(sess.run(summary_op), step)
                print("The summary at step = %d is saved" % step)
        else:
            raise ValueError("unexpected lr_policy")

    def check_is_nan(saver, sess, summary_writer, summary_op, batch_loss, step, config):
        if np.isnan(batch_loss):
            saver.save(sess, os.path.join(config.ckpt_dir, "model_step"), global_step=step, write_meta_graph=False)
            summary_writer.add_summary(sess.run(summary_op), step)
            raise ValueError("Model diverged with loss = NaN")

    def train_step(sess, train_op, loss, data_init, saver, graph, config):
        tf_img = tf.concat(tf.get_collection("image"), 0)
        tf_seg = tf.concat(tf.get_collection("label"), 0)
        tf_logit = tf.concat(tf.get_collection("get_logit"), 0)
        tf_prob = tf.nn.softmax(tf_logit)
        tf_monitor = tf.get_collection("monitor")
        tf_monitor_grad = tf.get_collection("monitor_grad")
        tf_lr = tf.squeeze(tf.concat(tf.get_collection("lr"), 0))
        global_step = tf.squeeze(tf.get_collection("global_step"))

        # calculate miou
        tf_pred = tf.argmax(tf_logit, 3)
        tf_pred_onehot = tf.one_hot(tf.cast(tf_pred, tf.int32), config.num_classes)
        onehot_gt = tf.one_hot(tf.cast(tf.squeeze(tf_seg, 3), tf.int32), config.num_classes)
        tf_intersection = tf_pred_onehot * onehot_gt
        tf_union = tf_pred_onehot + onehot_gt - tf_intersection
        tf_iou = tf.reduce_sum(tf_intersection, [0, 1, 2]) / tf.reduce_sum(tf_union, [0, 1, 2])
        tf_miou = tf.reduce_mean(tf_iou)
        tf_diminish = tf.get_collection("diminish")

        tf.summary.scalar("miou", tf_miou)
        tf.summary.scalar("batch_size", config.batch_size)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=config.ckpt_dir, graph=graph)

        should_continue = True
        print("Start training...")
        step = sess.run(global_step)
        sess.run(data_init)
        while should_continue:
            try:
                start_time = time.time()
                if config.do_grad_aggregation:
                    raise ValueError("this coding block much be debugged first. Check train_op")
                    gradients_to_be_aggregated = []
                    total_loss_record = []

                    for schedule in config.grad_aggregation_schedule:
                        if schedule[0] >= step:
                            num_grad_aggregation = schedule[1]
                            break
                    miou_agg = 0
                    for i in range(num_grad_aggregation):
                        loss, gradients, miou = sess.run([train_op[0], train_op[1], tf_miou])
                        total_loss_record.append(loss)
                        gradients_to_be_aggregated.append(gradients)
                        miou_agg += miou
                    batch_loss = np.mean(total_loss_record)
                    feed_dict = dict()
                    for i, ph in enumerate(train_op[2]):
                        feed_dict[ph] = np.stack([g[i] for g in gradients_to_be_aggregated], axis=0).mean(axis=0)
                    step, _ = sess.run([global_step, train_op[3]], feed_dict=feed_dict)
                else:
                    if tf_diminish:
                        _, batch_loss, step, miou, monitor, monitor_grad, diminish = sess.run(
                            [train_op, loss, global_step, tf_miou, tf_monitor, tf_monitor_grad, tf_diminish])
                    else:
                        _, batch_loss, step, miou, monitor, monitor_grad, lr = sess.run([train_op, loss, global_step, tf_miou, tf_monitor, tf_monitor_grad, tf_lr])
            except:
                sess.run(data_init)
                ##########################################################################
            # # save_statistics of feature map in "monitor" variable
            # if step <= config.slow_step_size:
            #     w_var = open(os.path.join(config.ckpt_dir, "00.statistics_var.csv"), "a+")
            #     w_grad_std = open(os.path.join(config.ckpt_dir, "00.statistics_grad_std.csv"), "a+")
            #     log_var = w_var.readlines()
            #     log_w_grad_std = w_grad_std.readlines()
            #
            #     if not log_var:
            #         layer_names = []
            #         for m in monitor:
            #             if len(m.keys()) != 1:
            #                 raise ValueError("unexpected")
            #             layer_names.append(m.keys())
            #         w_var.write("step, ")
            #         w_var.write(", ".join([str(l) for l in layer_names]) + "\n")
            #     var_container = []
            #     for m in monitor:
            #         var_container.append(m.values()[0])  # variance
            #     w_var.write("%s," % step)
            #     w_var.write(", ".join([str(vv) for vv in var_container]) + "\n")
            #
            #     if not log_w_grad_std:
            #         layer_names = []
            #         for m in monitor_grad:
            #             if len(m.keys()) != 1:
            #                 raise ValueError("unexpected")
            #             layer_names.append(m.keys())
            #         w_grad_std.write("step, ")
            #         w_grad_std.write(", ".join([str(l) for l in layer_names]) + "\n")
            #     w_grad_std_container = []
            #     for m in monitor_grad:
            #         w_grad_std_container.append(m.values()[0])  # standard deviation
            #     w_grad_std.write("%s," % step)
            #     w_grad_std.write(", ".join([str(vv) for vv in w_grad_std_container]) + "\n")
            ##############################################################################

            should_continue = False if step >= config.max_step else True
            elapsed = time.time() - start_time

            # assert not np.isnan(batch_loss), "Model diverged with loss = NaN"
            check_is_nan(saver, sess, summary_writer, summary_op, batch_loss, step, config)

            if step % config.log_steps == 0:
                print("step=%d(%.3f sec/step), total loss=%.3f, miou=%.3f, lr=%.9f" % (step, elapsed, batch_loss, miou, lr))
                # print("step=%d(%.3f sec/step), miou_loss=%.3f " % (step, elapsed, batch_loss))
            # save checkpoint and summary at every certain interval
            save_checkpoint_and_summary(saver, sess, summary_writer, summary_op, step, config)
            if config.lr_policy == "cyclical":
                if step == config.cycle_step_size:
                    sess.run(data_init)

    def start_train(train_tensor, loss, data_init, hvd, config):
        saver = tf.train.Saver(max_to_keep=5000)
        graph = tf.get_default_graph()
        with graph.as_default() as graph:
            global_init_fn = tf.global_variables_initializer()
            local_init_fn = tf.local_variables_initializer()
            init_fn = tf.group(global_init_fn, local_init_fn)
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            session_config.allow_soft_placement = True
            session_config.gpu_options.visible_device_list = str(hvd.local_rank())
            with tf.Session(config=session_config) as sess:
                all_ckpt_list = get_all_ckpt_list(config)
                if all_ckpt_list:  # assumed the current model is intended to continue training if latest checkpoint exists
                    print("=============================== Attention ===============================")
                    print("Training will be continued from the last checkpoint...")
                    saver.restore(sess, all_ckpt_list[-1])
                    sess.run(hvd.broadcast_global_variables(0))
                    print("The last checkpoint is loaded!")
                else:
                    sess.run(init_fn)
                    sess.run(hvd.broadcast_global_variables(0))
                    if config.pretrained_ckpt_dir:  # restore pretrained model
                        print("=============================== Attention ===============================")
                        print("Training will be started using the specified pretrained model...")
                        restore_pretrained_model_developing(sess, config, False)
                        print("The pretrained model is loaded!")
                    else:
                        print("=============================== Attention ===============================")
                        print("Training will be started from scratch...")
                train_step(sess, train_tensor, loss, data_init, saver, graph, config)
                print("=============================== Attention ===============================")
                print("Training is done!")
