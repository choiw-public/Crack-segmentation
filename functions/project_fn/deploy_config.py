import importlib as imp
from bunch import Bunch
import os


def deploy(args):
    phase = args.phase
    config_path = ".".join(["configs", "config_%s" % phase])
    config = imp.import_module(config_path).config
    config["ckpt_dir"] = "/".join(["./model", "checkpoints"])
    config['phase'] = phase
    if phase == "train":
        config["is_train"] = True
        os.makedirs(config["ckpt_dir"], exist_ok=True)
    elif phase == "eval":
        config["is_train"] = False
        config["efficient"] = False
        config["weight_decay"] = None
        config["blur_dir"] = None
        config["background_dir"] = None
        config["bnorm_trainable"] = None
        config["eval_log_dir"] = "/".join(["./model", "eval_metric"])
        os.makedirs(config["eval_log_dir"], exist_ok=True)
    elif phase == "vis":
        config["batch_size"] = 1
        config["efficient"] = False
        config["is_train"] = False
        config["weight_decay"] = None
        config["background_dir"] = None
        config["bnorm_trainable"] = None
        config["vis_result_dir"] = "/".join(["./model", "vis_results"])
        os.makedirs(config["vis_result_dir"], exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["physical_gpu_id"])
    config = Bunch(config)
    return config
