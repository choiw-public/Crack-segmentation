# evaluation Config
config = {
    "physical_gpu_id": 0,
    "dtype": "fp16",
    "num_classes": 2,
    "ckpt_start": 1,
    "ckpt_end": 1,
    "ckpt_step": 1,
    "img_step": 1,
    "img_dir": "./datasets/all_newV2_raw/img",
    "seg_dir": "./datasets/all_newV2_raw/seg",
    "eval_log_dir": "evaluation",
    "batch_size": 1,
}
