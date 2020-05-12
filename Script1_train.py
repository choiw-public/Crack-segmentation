from functions.project_fn.deploy_config import DeployConfig
from functions.project_fn.input_pipeline import build_input_pipeline

model_name = 'model_last_continue'
config = DeployConfig(model_name, 'train')

# # I don't know why but below two lines should be included
data, data_init = build_input_pipeline(config)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
with tf.Session() as sess:
    sess.run(data_init)
    img = sess.run(data['input'])

from functions.project_fn.model_utils_developing import build_model
from functions.project_fn.model_utils_developing import start_train

train_tensor, loss, hvd = build_model(data, config)
# start train
start_train(train_tensor, loss, data_init, hvd, config)
