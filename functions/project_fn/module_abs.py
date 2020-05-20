from functions.project_fn.utils import get_shape as get_shape
import tensorflow as tf


class Module:
    def get_kernel(self, target_tensor, kernel_size, kernel_depth, transpose=False):
        in_channel = get_shape(target_tensor)[-1]
        if transpose:
            kernel_shape = [kernel_size, kernel_size, kernel_depth, in_channel]
        else:
            kernel_shape = [kernel_size, kernel_size, in_channel, kernel_depth]

        if self.weight_decay:
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
        else:
            regularizer = None
        return tf.get_variable("kernel", kernel_shape, self.dtype, tf.initializers.he_uniform(), regularizer, True)

    def conv_block(self, tensor_in, kernel_size, stride, out_depth):
        def build(main_pipe):
            kernel = self.get_kernel(main_pipe, kernel_size, out_depth)
            main_pipe = tf.nn.conv2d(main_pipe, kernel, [1, stride, stride, 1], "SAME")
            main_pipe = tf.layers.batch_normalization(main_pipe, training=self.is_train, fused=True)
            main_pipe = tf.nn.elu(main_pipe)
            return main_pipe

        if self.is_train:
            build = tf.contrib.layers.recompute_grad(build)
        return build(tensor_in)

    def transpose_conv_block(self, tensor_in, kernel_size, stride, out_depth, out_shape):
        def build(main_pipe):
            kernel = self.get_kernel(main_pipe, kernel_size, out_depth, transpose=True)
            main_pipe = tf.nn.conv2d_transpose(main_pipe, kernel, out_shape, [1, stride, stride, 1], 'SAME')
            main_pipe = tf.layers.batch_normalization(main_pipe, training=self.is_train, fused=True)
            main_pipe = tf.nn.elu(main_pipe)
            return main_pipe

        if self.is_train:
            build = tf.contrib.layers.recompute_grad(build)
        return build(tensor_in)

    def gc_block(self, tensor_in, factor, scope='gc_block'):
        # GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
        def build(main_pipe):
            with tf.variable_scope(scope):
                with tf.variable_scope('context'):
                    n, h, w, c = get_shape(main_pipe)
                    tensor_in_flatten = tf.reshape(main_pipe, [n, h * w, c])
                    kernel = self.get_kernel(main_pipe, 1, 1)
                    context = tf.nn.conv2d(main_pipe, kernel, strides=[1, 1, 1, 1], padding='SAME')
                    context = tf.reshape(context, [n, h * w, 1])
                    context = tf.nn.softmax(context, axis=1)
                    context = tf.matmul(tensor_in_flatten, context, transpose_a=True)
                    context = tf.reshape(context, [n, 1, 1, c])

                with tf.variable_scope('transform'):
                    with tf.variable_scope('shrink'):
                        kernel = self.get_kernel(context, 1, int(c / factor))
                        transform = tf.nn.conv2d(context, kernel, [1, 1, 1, 1], 'SAME')
                        transform = tf.contrib.layers.layer_norm(transform, center=True, scale=True, scope=scope)
                        transform = tf.nn.relu(transform)
                    with tf.variable_scope('expand'):
                        kernel = self.get_kernel(transform, 1, c)
                        transform = tf.nn.conv2d(transform, kernel, [1, 1, 1, 1], 'SAME')
                        transform = tf.nn.sigmoid(transform)
                return main_pipe + transform

        if self.is_train:
            build = tf.contrib.layers.recompute_grad(build)
        return build(tensor_in)

    def convolution(self, tensor_in, kernel_size, stride, out_depth, scope):
        with tf.variable_scope(scope):
            return self.conv_block(tensor_in, kernel_size, stride, out_depth)

    def squeezing_dense(self, tensor_in, depths, conv_size, strides, scope, do_gc=False, gc_factor=None):
        raise NotImplementedError("Full code will be shared in future")

    def shortcut(self, tensor_in, low_level, kernel_size, stride, out_depth, scope):
        with tf.variable_scope(scope):
            low_level = self.conv_block(low_level, kernel_size, stride, out_depth)
            return tf.concat([tensor_in, low_level], 3)

    def upscale(self, tensor_in, fp_feature, kernel_size, stride, out_depth, scope):
        raise NotImplementedError("Full code will be shared in future")

    def get_logit(self, tensor_in, kernel_size, stride):
        def build(main_pipe):
            kernel = self.get_kernel(main_pipe, kernel_size, self.num_classes)
            main_pipe = tf.nn.conv2d(main_pipe, kernel, [1, stride, stride, 1], 'SAME')
            return main_pipe

        with tf.variable_scope('get_logit'):
            if self.is_train:
                build = tf.contrib.layers.recompute_grad(build)
        return build(tensor_in)
