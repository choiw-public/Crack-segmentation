# evaluation Config
config = {
    "physical_gpu_id": 0,
    "dtype": "fp32",
    "num_classes": 2,
    "ckpt_start": 400000,
    "ckpt_end": 400000,
    "ckpt_step": 1,
    "img_step": 1,
    "img_dir": None,  # folder path of jpg images
    "seg_dir": None,  # folder path of ground truth
    "eval_log_dir": "evaluation",
}
