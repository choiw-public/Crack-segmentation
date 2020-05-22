# evaluation Config
config = {
    "physical_gpu_id": 0,
    "dtype": "fp32",
    "num_classes": 2,
    "ckpt_start": 400000,
    "ckpt_end": 400000,
    "ckpt_step": 1,
    "img_step": 1,
    "img_dir": "/media/wooram/data_hdd/00.DL_datasets/00.civil/deepcrack/images/test/jpg",
    "seg_dir": "/media/wooram/data_hdd/00.DL_datasets/00.civil/deepcrack/segmentations/test",
    "eval_log_dir": "evaluation",
}
