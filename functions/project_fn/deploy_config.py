from tensorflow.dtypes import float16, float32
import importlib as imp
from bunch import Bunch
import os


def deploy(model_name, phase):
    config_name = ".".join(["models", model_name, "configs", "config_%s" % phase])
    config = imp.import_module(config_name).config
    result_dir = "/".join(["./models", model_name, "results"])
    if phase == "train":
        config["is_train"] = True
        config["ckpt_dir"] = "/".join([result_dir, "saved_model"])
        os.makedirs(config["ckpt_dir"], exist_ok=True)
    elif phase == "eval":
        config["is_train"] = False
        config["background_dir"] = None
        config["bnorm_trainable"] = None
        config["eval_log_dir"] = "/".join([result_dir, "eval_metric"])
        os.makedirs(config["eval_log_dir"], exist_ok=True)
    elif phase == "vis":
        config["batch_size"] = 1
        config["is_train"] = False
        config["background_dir"] = None
        config["bnorm_trainable"] = None
        config["vis_result_dir"] = "/".join([result_dir, "vis_results"])
        os.makedirs(config["vis_result_dir"], exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["physical_gpu_id"])
    if config["dtype"] == "fp16":
        config["dtype"] = float16
    elif config["dtype"] == "fp32":
        config["dtype"] = float32
    else:
        raise ValueError("Unexpected dtype:%s" % config["dtype"])
    config = Bunch(config)
    return config
