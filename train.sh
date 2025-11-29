
# Default values
nproc_per_node=4
data_dir="/ariesdv0/tongxuan/Datasets/dynamics_split_data/tshirt_v1_dynamics_inter3_h03_n05"
per_gpu_batch_size=16
num_train_epochs=99999
checkpointing_steps=2000
checkpoints_total_limit=10
gradient_accumulation_steps=8
learning_rate=1e-5
lr_warmup_steps=1000
lr_scheduler="cosine"
seed=1
num_workers=16
mixed_precision="bf16"
allow_tf32=true
cond="rgbd"
exp_description="dynamics"
exp_tags=("dynamics")
scale_lr=false
exp_name="11_11_resume_dynamics_tshirt_larger_delta"
resume_from_checkpoint="latest"
pretrained_model_name_or_path="None"
# pretrained_model_name_or_path="/ariesdv0/tongxuan/Code/UniClothDiff/experiments/dynamics_tshirt_h03n05_occ/checkpoints/checkpoint-90000"


# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --nproc_per_node)
      nproc_per_node="$2"
      shift
      shift
      ;;
    --data_dir)
      data_dir="$2"
      shift # past argument
      shift # past value
      ;;
    --per_gpu_batch_size)
      per_gpu_batch_size="$2"
      shift
      shift
      ;;
    --num_train_epochs)
      num_train_epochs="$2"
      shift
      shift
      ;;
    --checkpointing_steps)
      checkpointing_steps="$2"
      shift
      shift
      ;;
    --checkpoints_total_limit)
      checkpoints_total_limit="$2"
      shift
      shift
      ;;
    --gradient_accumulation_steps)
      gradient_accumulation_steps="$2"
      shift
      shift
      ;;
    --learning_rate)
      learning_rate="$2"
      shift
      shift
      ;;
    --lr_warmup_steps)
      lr_warmup_steps="$2"
      shift
      shift
      ;;
    --lr_scheduler)
      lr_scheduler="$2"
      shift
      shift
      ;;
    --seed)
      seed="$2"
      shift
      shift
      ;;
    --num_workers)
      num_workers="$2"
      shift
      shift
      ;;
    --mixed_precision)
      mixed_precision="$2"
      shift
      shift
      ;;
    --output_dir)
      output_dir="$2"
      shift
      shift
      ;;
    --allow_tf32)
      allow_tf32=true
      shift
      ;;
    --cond)
      cond="$2"
      shift
      shift
      ;;
    --exp_description)
      exp_description="$2"
      shift
      shift
      ;;
    --exp_tags)
      IFS=',' read -r -a exp_tags <<< "$2"
      shift
      shift
      ;;
    --scale_lr)
      scale_lr=true
      shift
      ;;
    --exp_name)
      exp_name="$2"
      shift
      shift
      ;;
    --resume_from_checkpoint)
      resume_from_checkpoint="$2"
      shift
      shift
      ;;
    --pretrained_model_name_or_path)
      pretrained_model_name_or_path="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

# Construct the command
command="/root/anaconda3/envs/clothdiff/bin/python -m torch.distributed.run \
--nproc_per_node=${nproc_per_node} \
--nnodes=1 \
--node_rank=0 train.py \
--data_dir=${data_dir} \
--per_gpu_batch_size=${per_gpu_batch_size} \
--num_train_epochs=${num_train_epochs} \
--checkpointing_steps=${checkpointing_steps} \
--checkpoints_total_limit=${checkpoints_total_limit} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--learning_rate=${learning_rate} \
--lr_warmup_steps=${lr_warmup_steps} \
--lr_scheduler=${lr_scheduler} \
--seed=${seed} \
--num_workers=${num_workers} \
--mixed_precision=${mixed_precision} \
--output_dir=\"${output_dir}\" \
--exp_name=\"${exp_name}\" \
--resume_from_checkpoint=\"${resume_from_checkpoint}\" \
--pretrained_model_name_or_path=\"${pretrained_model_name_or_path}\" \
--config="configs/train_dyn_tshirt_interp_dec.yaml" \
"

# Add optional flags
if [ "${allow_tf32}" = true ]; then
  command+="--allow_tf32 "
fi

command+="--cond \"${cond}\" \
--exp_tags ${exp_tags[@]} \
"

if [ "${scale_lr}" = true ]; then
  command+="--scale_lr"
fi

# Run the command
echo "Running command:"
echo ${command}
eval ${command}
