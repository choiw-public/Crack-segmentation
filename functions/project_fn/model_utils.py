from functions.project_fn.preprocess_developing import normalize_input
from functions.project_fn.misc_utils import get_all_ckpt_list
from functions.project_fn import loss_functions
import horovod.tensorflow as hvd
from math import pi
import tensorflow as tf
import numpy as np
import imp
import os
import collections
import time

Clone = collections.namedtuple("Clone",
                               ["logit",  # Whatever model_fn() returned.
                                "scope",  # The scope used to create it.
                                "device",  # The device used to create.
                                ])


def get_loss_fn(loss_fn_name):
    if loss_fn_name == "miou":
        return loss_functions.miou_loss
    elif loss_fn_name == "focal":
        return loss_functions.focal_loss
    elif loss_fn_name == "focal_miou":
        return loss_functions.focal_miou_loss
    elif loss_fn_name == "weighted_miou":
        return loss_functions.weighted_miou_loss
    elif loss_fn_name == "f1":
        return loss_functions.mf1_loss
    elif loss_fn_name == "mf1_miou":
        return loss_functions.mf1_miou_loss
    elif loss_fn_name == "miou_recall_loss":
        return loss_functions.miou_recall_loss
    elif loss_fn_name == "xntropy":
        return loss_functions.xntropy_loss
    elif loss_fn_name == "mse":
        return loss_functions.l2_loss
    elif loss_fn_name == "l1":
        return loss_functions.l1_loss
    elif loss_fn_name == "pseudo_huber":
        return loss_functions.pseudo_huber_loss
    elif loss_fn_name == "ridge_pseudo_huber":
        return loss_functions.ridge_pseudo_huber_loss
    elif loss_fn_name == "ms_ssim_lcs_random_gaussian_pseudo_huber":
        return loss_functions.ms_ssim_lcs_term_loss_with_random_gaussian_pseudo_huber
    elif loss_fn_name == "ridge_l1":
        return loss_functions.ridge_l1_loss
    elif loss_fn_name == "ssim_lcs_gaussian":
        return loss_functions.ssim_lcs_term_loss_with_gaussian
    elif loss_fn_name == "rec_mse":  # this function is shit
        return loss_functions.rectified_mse_loss
    elif loss_fn_name == "ssim_box_l":
        return loss_functions.ssim_l_term_loss_with_box_kernel
    elif loss_fn_name == "ssim_box_cs":
        return loss_functions.ssim_cs_term_loss_with_box_kernel
    elif loss_fn_name == "ssim_box_lcs":
        return loss_functions.ssim_lcs_term_loss_with_box_kernel
    elif loss_fn_name == "ssim_box_lcs_l1":
        return loss_functions.ssim_lcs_term_l1_loss_with_box_kernel
    elif loss_fn_name == "ms_ssim_cs_random_gaussian":
        return loss_functions.ms_ssim_cs_term_loss_with_random_gaussian
    elif loss_fn_name == "ms_ssim_lcs_random_gaussian":
        return loss_functions.ms_ssim_lcs_term_loss_with_random_gaussian
    elif loss_fn_name == "ms_ssim_cs_r_random_gaussian":
        return loss_functions.ms_ssim_cs_term_and_ridge_term_with_random_gaussian
    elif loss_fn_name == "sobel":
        return loss_functions.sobel_loss
    else:
        raise ValueError("unsupported loss function.")


def get_learning_rate(config, global_step):
    global_step = tf.cast(global_step, tf.float64)
    const_0 = tf.constant(0, dtype=tf.float64)
    const_1 = tf.constant(1, dtype=tf.float64)
    const_2 = tf.constant(2, dtype=tf.float64)
    if config.lr_policy == "native":
        return config.lr
    elif config.lr_policy == "slow_start":
        step_size = tf.constant(config.slow_step_size, tf.float64)
        return tf.reduce_min([config.lr * (global_step + const_1) / step_size, config.lr])
    elif config.lr_policy == "poly":
        return tf.train.polynomial_decay(tf.constant(config.start_lr, tf.float64),
                                         global_step,
                                         tf.constant(config.max_step, tf.float64),
                                         end_learning_rate=tf.constant(config.end_lr, tf.float64),  # 0.0
                                         power=tf.constant(config.power, tf.float64))

    elif config.lr_policy == "cyclical":
        step_size = tf.constant(config.cycle_step_size, tf.float64)
        max_lr = tf.constant(config.max_lr, tf.float64)
        min_lr = tf.constant(config.min_lr, tf.float64)
        gamma = tf.constant(config.gamma, tf.float64)
        cycle = tf.floor(const_1 + global_step / (const_2 * step_size))
        x = tf.abs(global_step / step_size - const_2 * cycle + const_1)
        clr = (max_lr - min_lr) * tf.maximum(const_0, const_1 - x)
        if config.cyclical_mode == "triangular":
            pass
        elif config.cyclical_mode == "triangular2":
            clr = clr / (const_2 ** (cycle - const_1))
        elif config.cyclical_mode == "exp_range":
            clr = clr * gamma ** global_step
        elif config.cyclical_mode == "cosine":
            max_lr_decay_step = tf.floor(const_1 + global_step / step_size)
            max_lr_decay = tf.constant(config.max_lr_decay, tf.float64)
            max_lr = max_lr * (max_lr_decay ** (max_lr_decay_step - const_1))
            cos_inner = (tf.constant(pi, tf.float64) * tf.floormod(global_step, step_size)) / step_size
            return (max_lr - min_lr) / const_2 * (tf.cos(cos_inner) + const_1) + min_lr
        else:
            raise ValueError("unsupported cyclical_mode")
        # polynomial cyclicical learning rate in exponential range
        # clr = clr + ((1 - global_step / config.max_step) ** 0.4)
        return min_lr + clr
    elif config.lr_policy == "fixed":
        return config.lr
    else:
        raise ValueError("unsupported")


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
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


def gradients_with_loss_scaling(loss, variables, loss_scale):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    return [grad / loss_scale for grad in tf.gradients(loss * loss_scale, variables)]


def build_train_op(optimizer, loss, global_step, config):
    all_trainable_variables = tf.trainable_variables()
    if config.layers_to_only_be_trained:
        layers_to_only_be_trained = config.layers_to_only_be_trained
        target_variables = []
        for layer in layers_to_only_be_trained:
            target_variables += [var for var in all_trainable_variables if layer in var.name]
    else:
        target_variables = all_trainable_variables
    grads_and_vars = optimizer.compute_gradients(loss, var_list=target_variables)

    for index, grad in enumerate(grads_and_vars):
        tf.summary.histogram("{}-grad".format(grads_and_vars[index][1].name), grads_and_vars[index][0])
        tf.summary.histogram(grads_and_vars[index][1].name, grads_and_vars[index][1])

    none_grad_vars = []
    for grad, var in grads_and_vars:
        if grad is None:
            none_grad_vars.append(var)
    if none_grad_vars:
        for var in none_grad_vars:
            print(var.name)
        raise ValueError("The above variables have no gradient")

    if config.layers_to_be_multiplied:
        grad_mult = get_model_gradient_multipliers(config.layers_to_be_multiplied, config.gradient_multiplier)
        grads_and_vars = tf.contrib.training.multiply_gradients(grads_and_vars, grad_mult)
    return optimizer.apply_gradients(grads_and_vars, global_step=global_step)


def build_model(data, config):
    hvd.init()
    architecture_fn = imp.load_source("build_architecture",
                                      os.path.join(config.model_config_dir, "build_architecture.py")).build_architecture
    in_data = data["input"]
    gt_data = data["gt"]
    if config.phase == "train":
        # data = build_input_pipeline(gpu_id, config)
        tf.add_to_collection("input", in_data)
        print("Deploying model to GPU:%d..." % config.physical_gpu_id)
        with tf.device("/GPU:0"), tf.variable_scope("fp32_var", custom_getter=fp32_var_getter, use_resource=True, reuse=False):
            logits = architecture_fn(normalize_input(in_data), config)
            loss_fn = get_loss_fn(config.loss_fn_name)
            # loss + weight_decay loss
            loss = loss_fn(logits, gt_data, config)
            l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if l2_loss:
                loss += tf.add_n(l2_loss)
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.cast(get_learning_rate(config, global_step), tf.float32)
            tf.summary.scalar("learning_rate", learning_rate)
            tf.add_to_collection("lr", learning_rate)

            if config.dtype == tf.float16:
                epsilon = 1e-4
            elif config.dtype == tf.float32:
                epsilon = 1e-8
            else:
                raise ValueError("unexpected")
            # optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=config.weight_decay, learning_rate=learning_rate, epsilon=epsilon)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # train_op = optimizer.apply_gradients(zip(grads, variables), global_step=global_step)

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

            if config.dtype == tf.float16:
                loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(128, 100)
                # Wraps the original optimizer in a LossScaleOptimizer.
                optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
                compression = hvd.Compression.fp16
            elif config.dtype == tf.float32:
                compression = hvd.Compression.none
            else:
                raise ValueError("unexpected dtype")
            optimizer = hvd.DistributedOptimizer(optimizer, compression=compression)

            tf.add_to_collection("logit", logits)
            tf.add_to_collection("label", gt_data)
            tf.add_to_collection("losses", loss)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = build_train_op(optimizer, loss, global_step, config)
                # train_op = optimizer.minimize(loss, global_step=global_step)

            print("Deploying model is done!")
            return train_op, loss, hvd
    elif config.phase == "eval":
        with tf.device("/GPU:0"), tf.variable_scope("fp32_var", custom_getter=fp32_var_getter, use_resource=True, reuse=False):
            logits = architecture_fn(normalize_input(in_data), config)
        pred = tf.expand_dims(tf.argmax(logits, 3), 3)
        confusion_matrix = tf.confusion_matrix(tf.reshape(gt_data, [-1]), tf.reshape(pred, [-1]), config.num_classes, dtype=tf.float32)
        return confusion_matrix, hvd

    elif config.phase == "vis":
        with tf.device("/GPU:0"), tf.variable_scope("fp32_var", custom_getter=fp32_var_getter, use_resource=True, reuse=False):
            logits = architecture_fn(normalize_input(in_data), config)
        pred = tf.squeeze(tf.argmax(logits, 3))
        return pred, hvd


def get_model_gradient_multipliers(target_tensor_scope, gradient_multiplier):
    """Gets the gradient multipliers.

    The gradient multipliers will adjust the learning rates for model
    variables. For the task of semantic segmentation, the models are
    usually fine-tuned from the models trained on the task of image
    classification. To fine-tune the models, we usually set larger (e.g.,
    10 times larger) learning rate for the parameters of last layer.

    Args:
      target_tensor_scope: Scopes of layers to be multiplied by "gradient_multiplier"
      gradient_multiplier: The gradient multiplier for last layers.

    Returns:
      The gradient multiplier map with variables as key, and multipliers as value.
    """
    gradient_multipliers = {}
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        # Double the learning rate for biases.
        if "bias" in var.op.name:
            gradient_multipliers[var.op.name] = 2.

        # Use larger learning rate for last layer variables.
        for layer in target_tensor_scope:
            if layer in var.op.name and "bias" in var.op.name:
                gradient_multipliers[var.op.name] = 2 * gradient_multiplier
                break
            elif layer in var.op.name:
                gradient_multipliers[var.op.name] = gradient_multiplier
                break
    return gradient_multipliers


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


def restore_pretrained_model_developing(sess, config, ignore_missing_vars=True):
    print("Loading a 'PRE-TRAINED' model from %s", config.pretrained_ckpt_dir)
    # Variables that will not be restored.
    exclude_list = ["global_step"]
    if config.layers_to_be_not_restored:
        raise ValueError("this has not been debugged yet")
        exclude_list.extend([config.task + "/" + exclude_name for exclude_name in config.layers_to_be_not_restored])

    variables_to_restore = get_variables_to_restore(exclude_list)

    if not variables_to_restore:
        raise ValueError("var_list cannot be empty")
    if ignore_missing_vars:
        reader = tf.pywrap_tensorflow.NewCheckpointReader(config.pretrained_ckpt_dir)
        if isinstance(variables_to_restore, dict):
            var_dict = variables_to_restore
        else:
            var_dict = {var.op.name: var for var in variables_to_restore}
        available_vars = {}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var] = var_dict[var]
            else:
                tf.logging.warning(
                    "Variable %s missing in checkpoint %s", var, config.pretrained_ckpt_dir)
        variables_to_restore = available_vars
    if variables_to_restore:
        excluded_vars = list(set(tf.global_variables()) - set(variables_to_restore))
        if excluded_vars:
            print("=============================== Attention ===============================")
            print("The following variables will not be restored from pretrained model:")
            for variable in excluded_vars:
                print("     %s" % variable)
            print("==========================================================================")
            print("\n")

        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, config.pretrained_ckpt_dir)
    else:
        raise NameError("no variables to restores")


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
    tf_img = tf.concat(tf.get_collection("input"), 0)
    tf_seg = tf.concat(tf.get_collection("label"), 0)
    tf_logit = tf.concat(tf.get_collection("logit"), 0)
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
