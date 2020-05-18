from functions.project_fn.deploy_config import deploy
from functions.project_fn.data_pipeline import DataPipeline
from functions.project_fn.model_utils import ModelHandler
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

model_name = "model_last_continue"
config = deploy(model_name, "train")

data_pipeline = DataPipeline(config)
model = ModelHandler(data_pipeline, config)






# from functions.project_fn.model_utils import build_model
# from functions.project_fn.model_utils import _start_train

# train_tensor, loss, hvd = build_model(data_pipeline, config)
# start _train_phase
# _start_train(train_tensor, loss, data_init, hvd, config)
