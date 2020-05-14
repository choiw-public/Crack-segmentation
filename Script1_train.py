from functions.project_fn.deploy_config import deploy
from functions.project_fn.input_pipeline_dev import InputPipeline
import tensorflow as tf

model_name = "model_last_continue"
config = deploy(model_name, "train")

# # I don"t know why but below two lines should be included
input_pipe = InputPipeline(config)
input_pipe.build()
tf_img = input_pipe.data['input']
tf_gt = input_pipe.data['gt']

with tf.Session() as sess:
    img, gt = sess.run([tf_img, tf_gt])

from functions.project_fn.model_utils import build_model
from functions.project_fn.model_utils import start_train

train_tensor, loss, hvd = build_model(data, config)
# start train
start_train(train_tensor, loss, data_init, hvd, config)
