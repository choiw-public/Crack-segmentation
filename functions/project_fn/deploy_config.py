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
    else:
        config["is_train"] = False
        config["blur_dir"] = None
        config["background_dir"] = None
        config["eval_log_dir"] = "/".join(["./model", "eval_metric"])
        config["batch_size"] = 1
        if phase == "eval":
            config["eval_log_dir"] = "/".join(["./model", "eval_metric"])
            os.makedirs(config["eval_log_dir"], exist_ok=True)
        elif phase == "vis":
            config["img_dir"] = config["data_dir"]
            if config["data_type"] == "image":
                config["vis_result_dir"] = "/".join(["./model", "vis_results", "image"])
            elif config["data_type"] == "video":
                config["vis_result_dir"] = "/".join(["./model", "vis_results", "video"])
            os.makedirs(config["vis_result_dir"], exist_ok=True)
        else:
            raise ValueError('Unexpected phase')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["physical_gpu_id"])
    config = Bunch(config)
    return config
