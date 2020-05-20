from functions.project_fn.deploy_config import deploy
from functions.project_fn.data_pipeline import DataPipeline
from functions.project_fn.model_handler import ModelHandler
import argparse
import tensorflow as tf

argparser = argparse.ArgumentParser()
argparser.add_argument('--phase', type=str, default='eval', help='options: train, eval, vis')
args = argparser.parse_args()

config = deploy(args)

data_pipeline = DataPipeline(config)
ModelHandler(data_pipeline, config)
