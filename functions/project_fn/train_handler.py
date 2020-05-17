from functions.project_fn.utils import list_getter
import re
import tensorflow as tf
import glob
from math import pi


class TrainHandler:
    def __init__(self):
        self.all_ckpt_list = [_.split(".")[0] for _ in list_getter(self.ckpt_dir, 'index')]

    def _build_train_op(self):
        self.grads_and_vars = self.optm_op.compute_gradients(self.loss, var_list=tf.trainable_variables())

        # gradient sanity check
        none_grad_vars = []
        for grad, var in self.grads_and_vars:
            if grad is None:
                none_grad_vars.append(var)
        if none_grad_vars:
            for var in none_grad_vars:
                print(var.name)
            raise ValueError('The above variables have no gradient')
        self.optm_op.apply_gradients(self.grads_and_vars, global_step=self.global_step)

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
        global_step = tf.cast(tf.train.get_or_create_global_step(), tf.float64)
        const_0 = tf.constant(0.0, dtype=tf.float64)
        const_1 = tf.constant(1, dtype=tf.float64)
        const_2 = tf.constant(2, dtype=tf.float64)
        slow_step_size = tf.constant(self.slow_step_size, tf.float64)
        cycle_step_size = tf.constant(self.cycle_step_size, tf.float64)

        max_lr = tf.constant(self.max_lr, tf.float64)
        min_lr = tf.constant(self.min_lr, tf.float64)
        max_lr_decay_step = tf.cond(tf.less_equal(global_step, slow_step_size),
                                    lambda: const_0,
                                    lambda: tf.floor(const_1 + (global_step - slow_step_size) / cycle_step_size))

        max_lr_decay = tf.constant(self.max_lr_decay, tf.float64)
        max_lr = max_lr * (max_lr_decay ** (max_lr_decay_step - const_1))
        cos_inner = (tf.constant(pi, tf.float64) * tf.floormod(global_step - slow_step_size, cycle_step_size)) / cycle_step_size

        self.lr = tf.cond(tf.less_equal(global_step, slow_step_size),
                          lambda: self.max_lr / slow_step_size * global_step,
                          lambda: (max_lr - min_lr) / const_2 * (tf.cos(cos_inner) + const_1) + min_lr)

    def _train_step(self, sess, graph):
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.ckpt_dir, graph=graph)

        should_continue = True
        print('Start training...')
        step = sess.run(self.global_step)
        while should_continue:
            try:
                start_time = time.time()
                _, batch_loss, step, miou, monitor, monitor_grad, lr = sess.run([train_op, loss, global_step, tf_miou, tf_monitor, tf_monitor_grad, tf_lr])
            except:
                sess.run(data_init)

            should_continue = False if step >= config.max_step else True
            elapsed = time.time() - start_time

            # assert not np.isnan(batch_loss), 'Model diverged with loss = NaN'
            check_is_nan(saver, sess, summary_writer, summary_op, batch_loss, step, config)

            if step % config.log_steps == 0:
                print('step=%d(%.3f sec/step), total loss=%.3f, miou=%.3f, lr=%.9f' % (step, elapsed, batch_loss, miou, lr))
                # print('step=%d(%.3f sec/step), miou_loss=%.3f ' % (step, elapsed, batch_loss))
            # save checkpoint and summary at every certain interval
            save_checkpoint_and_summary(saver, sess, summary_writer, summary_op, step, config)
            if config.lr_policy == 'cyclical':
                if step == config.cycle_step_size:
                    sess.run(data_init)

    def _start_train(self):
        saver = tf.train.Saver(max_to_keep=5000)
        graph = tf.get_default_graph()
        with graph.as_default() as graph:
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
            session_config.allow_soft_placement = True
            session_config.gpu_options.visible_device_list = str(self.hvd.local_rank())
            sess = tf.Session(config=session_config)
            if self.all_ckpt_list:  # assumed the current model is intended to continue training if latest checkpoint exists
                print('=============================== Attention ===============================')
                print('Training will be continued from the last checkpoint...')
                saver.restore(sess, self.all_ckpt_list[-1])
                sess.run(self.hvd.broadcast_global_variables(0))
                print('The last checkpoint is loaded!')
            else:
                global_init_fn = tf.global_variables_initializer()
                local_init_fn = tf.local_variables_initializer()
                init_fn = tf.group(global_init_fn, local_init_fn)
                sess.run(init_fn)
                sess.run(self.hvd.broadcast_global_variables(0))
                print('=============================== Attention ===============================')
                print('Training will be started from scratch...')
            self._train_step(sess, graph)
            print('=============================== Attention ===============================')
            print('Training is done!')

    def _train(self):
        self._miou_loss()
        l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if l2_loss:
            self.loss += tf.add_n(l2_loss)
        self._get_learning_rate()
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        if self.dtype == tf.float16:
            loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(128, 100)
            # Wraps the original optimizer in a LossScaleOptimizer.
            optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
            compression = self.hvd.Compression.fp16
        else:
            compression = self.hvd.Compression.none
        self.optm_op = self.hvd.DistributedOptimizer(optimizer, compression=compression)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._build_train_op()
            self._build_summary_op()
            self._start_train()
