# Train Config
config = {
    # GPU
    "dtype": "fp16",
    "physical_gpu_id": 0,
    "efficient": True,

    # optimization
    "num_classes": 2,

    # optimization - learning policy
    "slow_start_step_size": 2000,
    "cycle_step_size": 80000,
    "min_lr": 0.000001,
    "max_lr": 0.006,
    "max_lr_decay": 0.9,
    "max_step": 400001,
    "weight_decay": 0.00001,

    # logging
    "log_print_interval": 100,
    "ckpt_save_interval": 256,
    "summary_save_interval": 512,

    # input
    "dataset_dir": "./datasets/all_newV2_raw_aug/tfrecord",
    "background_dir": "./datasets/background",
    "background_proportion": 0.25,
    "blur_dir": None,  # './datasets/blur/tfrecord',
    "blur_proportion": 0.25,
    "batch_size": 2,

    # input - augmentation
    "random_scale_range": [0.8, 1.2],  # scale before cropping. None for skipping
    "crop_size": [384, 384],
    "flip_probability": 0.5,
    "rotate_probability": 0.5,
    "rotate_angle_by90": True,
    "rotate_angle_range": None,  # works only if "rotate_angle_by90: False"
    "random_quality_prob": 0.1,
    "random_quality": [30, 100],
    "rgb_permutation_prob": 0.5,
    "brightness_prob": 0.2,
    "brightness_constant": 0.3,
    "contrast_prob": 0.2,
    "contrast_constant": [1.0, 2.0],
    "hue_prob": 0.2,
    "hue_constant": [-0.3, 0.3],
    "saturation_prob": 0.2,
    "saturation_constant": [0.4, 3.0],
    "gaussian_noise_prob": 0.5,
    "gaussian_noise_std": [0.03, 0.1],
    "shred_prob": 0.0,
    "shred_piece_range": None,  # [max, min] number of shredded pieces
    "shred_shift_ratio": None,
    "shade_prob": 1.0,
    "shade_file": "./shades/shade.tfrecord",
    "warp_prob": 0.0,  # after 250000
    "warp_ratio": 0.4,
    "warp_crop_prob": 1.0,
    "elastic_distortion_prob": 0.0,  # not recommended for fine features
}
