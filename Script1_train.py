from functions.project_fn.deploy_config import deploy
from functions.project_fn.input_pipeline_dev import InputPipeline
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

model_name = "model_last_continue"
config = deploy(model_name, "eval")

# # I don"t know why but below two lines should be included
input_pipe = InputPipeline(config)

from functions.project_fn.model_utils import build_model
from functions.project_fn.model_utils import start_train

train_tensor, loss, hvd = build_model(data, config)
# start train
start_train(train_tensor, loss, data_init, hvd, config)
