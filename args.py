import argparse

class AppendToDefault(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, self.default) or []
        items.extend(values)
        setattr(namespace, self.dest, items)

def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    
    parser.add_argument(
        '--data_dir', 
        type=str, 
        help='Raw data directory.'
    )
    
    # Logging arguments
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='checkpoints/output/',
        help='Output directory.'
    )
    
    parser.add_argument(
        '--logging_dir', 
        type=str,
        default='logs/',
        help='Logging directory.'
    )
    
    # Accelerate arguments
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ', `"wandb"` (default) and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )   
    parser.add_argument(
        "--seed", 
        type=int, 
        default=259, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--use_ema", 
        action="store_true", 
        help="Whether to use EMA model."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        # default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--allow_tf32",
        # default=True, 
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=900000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="None",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resolution",
        default=(50, 50),
        help=(
            "Resolution of the cloth image."
        ),
    )
    parser.add_argument(
        "--cond",
        default='rgbd',
        type=str,
        choices=["rgb", "rgbd"],
        help=(
            "Condition mode. Choose between ['rgb', 'rgbd']."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=2000,
        help="Steps interval to save checkpoints.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=1000,
        help=(
            "Run sampling for visualization very XX optimization steps"
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=10,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="DEBUG",
        help=("Experiment description."),
    )
    parser.add_argument(
        "--exp_tags",
        nargs='+',
        action=AppendToDefault, 
        default=['clothdiff'],
        help=("Experiment tags."),
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=999999)
    parser.add_argument(
        "--depth_normalization", 
        default=False,
        action="store_true")
    parser.add_argument(
        "--config", 
        type=str,
        
        default="configs/train_state_est.yaml",
        
    )
    parser.add_argument(
        "--do_classifier_free_guidance",
        "-do_cfg", 
        default=True,
        action="store_true"
    )
    args = parser.parse_args()
    
    if args.resume_from_checkpoint == "None":
        args.resume_from_checkpoint = None
    if args.pretrained_model_name_or_path == "None":
        args.pretrained_model_name_or_path = None

    return args
