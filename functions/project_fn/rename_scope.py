import tensorflow as tf

Froms = ['exit', 'decoder1', 'decoder2', 'decoder3']
Tos = ['middle/block6', 'ASPP', 'decoder1', 'decoder2']
log_dir = './models/pretrained_DenSepNet'

checkpoint_list = tf.train.get_checkpoint_state(log_dir).all_model_checkpoint_paths

for ckpt in checkpoint_list:
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            variables = tf.contrib.framework.list_variables(ckpt)
            new_variables = []
            for i, (name, shape) in enumerate(variables):
                v = tf.contrib.framework.load_variable(ckpt, name)
                new_variable = None
                for (From, To) in zip(Froms, Tos):
                    if From in name:
                        new_variable = tf.Variable(v, name=name.replace(From, To))
                if new_variable:
                    new_variables.append(new_variable)
                else:
                    new_variables.append(tf.Variable(v, name=name))
            saver = tf.train.Saver(new_variables)
            sess.run(tf.global_variables_initializer())
            saver.save(sess, ckpt)
