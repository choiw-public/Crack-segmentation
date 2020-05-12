import tensorflow as tf
import imp
import os


class DeployConfig(object):
    def __init__(self, model_name, job):
        model_dir = os.path.join('./models', model_name)
        model_config_dir = os.path.join(model_dir, 'config')
        model_config = imp.load_source('configs', os.path.join(model_config_dir, 'config.py'))
        self.job = job
        self.log_root = 'results'
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_config_dir = model_config_dir
        tf.gfile.MkDir(os.path.join('./models', model_name, self.log_root))
        self.ckpt_dir = os.path.join('./models', model_name, self.log_root, 'saved_model')  # todo: change 'checkpoints' to 'saved_model'
        tf.gfile.MkDir(self.ckpt_dir)
        # configs ONLY for deblur task
        self.loss_fn_name = model_config.task['loss_fn_name']
        self.num_classes = model_config.task['num_classes']
        self.weight_decay = model_config.train['weight_decay']
        self.input_size = model_config.task['input_size']
        self.input_scale = model_config.task['input_scale']

        if job == 'train':
            self.is_train = True
            if model_config.train['dtype'] == 'fp16':
                self.dtype = tf.float16
            elif model_config.train['dtype'] == 'fp32':
                self.dtype = tf.float32
            else:
                raise ValueError('unexpected dtype')
            self.bnorm_trainable = model_config.train['bnorm_trainable']
            self.physical_gpu_id = model_config.train['physical_gpu_id']
            self.efficient = model_config.train['efficient']
            # logging
            self.log_steps = model_config.train['log_steps']
            self.ckpt_save_interval = model_config.train['ckpt_save_interval']
            self.summary_save_interval = model_config.train['summary_save_interval']
            # learnig rate policy
            self.lr_policy = model_config.train['lr_policy']
            self.slow_step_size = model_config.train['slow_step_size']
            if self.lr_policy == 'fixed':
                self.lr = model_config.train['lr']
            elif self.lr_policy == 'slow_start':
                self.lr = model_config.train['lr']
            elif self.lr_policy == 'poly':
                self.start_lr = model_config.train['start_lr']
                self.end_lr = model_config.train['end_lr']
                self.power = model_config.train['power']
            elif self.lr_policy == 'cyclical':
                self.cyclical_mode = model_config.train['cyclical_mode']
                self.cycle_step_size = model_config.train['cycle_step_size']
                self.min_lr = model_config.train['min_lr']
                self.max_lr = model_config.train['max_lr']
                self.max_lr_decay = model_config.train['max_lr_decay']
                self.gamma = model_config.train['gamma']
            self.max_step = model_config.train['max_step']
            self.batch_size = model_config.train['batch_size']
            self.do_grad_aggregation = model_config.train['do_grad_aggregation']
            self.grad_aggregation_schedule = model_config.train['grad_aggregation_schedule']

            # fine-tune
            self.pretrained_ckpt_dir = model_config.train['pretrained_ckpt_dir']
            self.gradient_multiplier = model_config.train['gradient_multiplier']
            self.layers_to_be_not_restored = model_config.train['layers_to_be_not_restored']
            self.layers_to_be_multiplied = model_config.train['layers_to_be_multiplied']
            self.layers_to_only_be_trained = model_config.train['layers_to_only_be_trained']
            # input
            self.dataset_dir = model_config.train['dataset_dir']
            self.background_dir = model_config.train['background_dir']
            self.background_proportion = model_config.train['background_proportion']
            # input preprocessing
            self.min_random_scale_factor = model_config.train['min_random_scale_factor']
            self.max_random_scale_factor = model_config.train['max_random_scale_factor']
            self.min_length_limit = model_config.train['min_length_limit']
            self.max_length_limit = model_config.train['max_length_limit']

            # augmentation
            self.flip_probability = model_config.train['flip_probability']
            self.rotate_probability = model_config.train['rotate_probability']
            self.rotate_angle_range = model_config.train['rotate_angle_range']
            self.rotate_angle_by90 = model_config.train['rotate_angle_by90']

            self.random_quality = model_config.train['random_quality']
            self.warp_prob = model_config.train['warp_prob']
            self.warp_ratio = model_config.train['warp_ratio']
            self.warp_crop_prob = model_config.train['warp_crop_prob']

            self.additional_augmentation_probability = model_config.train['additional_augmentation_probability']
            self.random_quality_prob = model_config.train['random_quality_prob']
            self.random_quality = model_config.train['random_quality']
            self.rgb_permutation_prob = model_config.train['rgb_permutation_prob']
            self.brightness_prob = model_config.train['brightness_prob']
            self.brightness_constant = model_config.train['brightness_constant']
            self.contrast_prob = model_config.train['contrast_prob']
            self.contrast_constant = model_config.train['contrast_constant']
            self.hue_prob = model_config.train['hue_prob']
            self.hue_constant = model_config.train['hue_constant']
            self.saturation_prob = model_config.train['saturation_prob']
            self.saturation_constant = model_config.train['saturation_constant']
            self.gaussian_noise_prob = model_config.train['gaussian_noise_prob']
            self.gaussian_noise_std = model_config.train['gaussian_noise_std']
            self.shade_prob = model_config.train['shade_prob']
            self.shade_source = model_config.train['shade_source']

            self.shred_vertical_prob = model_config.train['shred_vertical_prob']
            self.shred_horizontal_prob = model_config.train['shred_horizontal_prob']
            self.shred_num = model_config.train['shred_num']
            self.shift_ratio = model_config.train['shift_ratio']
            self.elastic_distortion_prob = model_config.train['elastic_distortion_prob']

            self.blur_proportion = model_config.train['blur_proportion']
            self.blur_dir = model_config.train['blur_dir']

            self.mixup_proportion = model_config.train['mixup_proportion']
            self.mixup_alpha = model_config.train['mixup_alpha']

        elif job == 'eval':
            self.is_train = False
            if model_config.evaluation['dtype'] == 'fp16':
                self.dtype = tf.float16
            elif model_config.evaluation['dtype'] == 'fp32':
                self.dtype = tf.float32
            else:
                raise ValueError('unexpected dtype')
            self.background_dir = None
            self.bnorm_trainable = False
            self.physical_gpu_id = model_config.evaluation['physical_gpu_id']
            self.ckpt_start = model_config.evaluation['ckpt_start']
            self.ckpt_end = model_config.evaluation['ckpt_end']
            self.ckpt_step = model_config.evaluation['ckpt_step']
            self.data_type = model_config.evaluation['data_type']
            if self.data_type == 'image':
                self.img_dir = model_config.evaluation['img_dir']
                self.seg_dir = model_config.evaluation['seg_dir']
            elif self.data_type == 'tfrecord':
                self.dataset_dir = model_config.evaluation['dataset_dir']
            else:
                raise ValueError('eval_data_type should be one of image or tfrecord')
            self.batch_size = model_config.evaluation['batch_size']
            self.eval_log_dir = os.path.join('./models', model_name, self.log_root, 'eval_metric')
            self.efficient = model_config.evaluation['efficient']
            tf.gfile.MkDir(self.eval_log_dir)
        elif job == 'vis':
            if model_config.visualization['dtype'] == 'fp16':
                self.dtype = tf.float16
            elif model_config.visualization['dtype'] == 'fp32':
                self.dtype = tf.float32
            else:
                raise ValueError('unexpected dtype')
            self.background_dir = None
            self.bnorm_trainable = False
            self.data_type = model_config.evaluation['data_type']
            if self.data_type == 'image':
                self.img_dir = model_config.evaluation['img_dir']
            elif self.data_type == 'tfrecord':
                self.dataset_dir = model_config.evaluation['dataset_dir']
            else:
                raise ValueError('eval_data_type should be one of image or tfrecord')
            self.batch_size = 1
            self.is_train = False
            self.physical_gpu_id = model_config.visualization['physical_gpu_id']
            self.ckpt_id = model_config.visualization['ckpt_id']
            self.img_dir = model_config.visualization['img_dir']
            self.vis_result_dir = os.path.join('./models', model_name, self.log_root, 'vis_results')
            self.efficient = model_config.visualization['efficient']
            tf.gfile.MkDir(self.vis_result_dir)
        elif job in ['combine', 'freeze']:
            self.is_train = False
            self.final_model_path = os.path.join(self.ckpt_dir, 'model_step-%d' % model_config.combine_or_freeze['final_model_id'])
            self.physical_gpu_id = [0]  # this line does nothing
        else:
            raise ValueError('not supported')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.physical_gpu_id)
