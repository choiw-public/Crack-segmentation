# visualization Config
config = {
    "physical_gpu_id": 0,
    "efficient": True,  # todo: check if this has any impact
    "dtype": "fp16",
    "num_classes": 2,
    "data_type": "tfrecord",
    "img_dir": "./datasets/all_newV2_raw/img",
    "dataset_dir": "'./datasets/all_newV2_raw/tfrecord'",
    "img_step": 1,
    "ckpt_id": 1,
    "vis_result_dir": "vis",
}
