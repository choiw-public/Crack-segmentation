"""AdaBound for Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re


class AdaBoundOptimizer(tf.train.Optimizer):
    """Optimizer that implements the AdaBound algorithm.

    See [Luo et al., 2019](https://openreview.net/forum?id=Bkg3g2R9FX)
    ([pdf](https://openreview.net/pdf?id=Bkg3g2R9FX)).
    """

    def __init__(self,
                 learning_rate=0.001,
                 final_lr=0.1,
                 beta1=0.9,
                 beta2=0.999,
                 gamma=1e-3,
                 epsilon=1e-8,
                 amsbound=False,
                 decay=0.,
                 weight_decay=0.,
                 exclude_from_weight_decay=None,
                 use_locking=False, name="AdaBound"):
        super(AdaBoundOptimizer, self).__init__(use_locking, name)

        if final_lr <= 0.:
            raise ValueError("Invalid final learning rate : {}".format(final_lr))
        if not 0. <= beta1 < 1.:
            raise ValueError("Invalid beta1 value : {}".format(beta1))
        if not 0. <= beta2 < 1.:
            raise ValueError("Invalid beta2 value : {}".format(beta2))
        if not 0. <= gamma < 1.:
            raise ValueError("Invalid gamma value : {}".format(gamma))
        if epsilon <= 0.:
            raise ValueError("Invalid epsilon value : {}".format(epsilon))

        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._final_lr = final_lr
        self._gamma = gamma
        self._epsilon = epsilon
        self._amsbound = amsbound
        self._decay = decay
        self._weight_decay = weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

        self._base_lr = learning_rate

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        lr = self._lr
        t = tf.cast(global_step, dtype=tf.float32)

        if self._decay > 0.:
            lr *= (1. / (1. + self._decay * t))

        t += 1

        bias_correction1 = 1. - (self._beta1 ** t)
        bias_correction2 = 1. - (self._beta2 ** t)
        step_size = (lr * tf.sqrt(bias_correction2) / bias_correction1)

        # Applies bounds on actual learning rate
        # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
        final_lr = self._final_lr * lr / self._base_lr
        lower_bound = final_lr * (1. - 1. / (self._gamma * t + 1.))
        upper_bound = final_lr * (1. + 1. / (self._gamma * t))

        assignments = []
        for grad, param in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adabound_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adabound_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            if self._amsbound:
                v_hat = tf.get_variable(
                    name=param_name + "/adabound_v_hat",
                    shape=param.shape.as_list(),
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())

            m_t = (
                    tf.multiply(self._beta1, m) + tf.multiply(1. - self._beta1, grad))
            v_t = (
                    tf.multiply(self._beta2, v) + tf.multiply(1. - self._beta2, tf.square(grad)))

            if self._amsbound:
                # Maintains the maximum of all 2nd moment running avg. till now
                v_hat_t = tf.maximum(v_hat, v_t)

                # Use the max. for normalizing running avg. of gradient
                denom = (tf.sqrt(v_hat_t) + self._epsilon)
            else:
                denom = (tf.sqrt(v_t) + self._epsilon)

            step_size_p = step_size * tf.ones_like(denom)
            step_size_p_bound = step_size_p / denom

            lr_t = m_t * tf.clip_by_value(t=step_size_p_bound,
                                          clip_value_min=lower_bound,
                                          clip_value_max=upper_bound)
            p_t = param - lr_t

            if self._do_use_weight_decay(param_name):
                p_t += self._weight_decay * param

            update_list = [param.assign(p_t), m.assign(m_t), v.assign(v_t)]
            if self._amsbound:
                update_list.append(v_hat.assign(v_hat_t))

            assignments.extend(update_list)

        # update the global step
        assignments.append(global_step.assign_add(1))

        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self._weight_decay:
            return False
        if self._exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    @staticmethod
    def _get_variable_name(param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
