export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0,1,2,3,4,5,6,7"

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("1e-4") #("1e-4" "1e-3")
LR_SCHEDULES=("cosine_with_restarts")
OPTIMIZERS=("adamw") #OPTIMIZERS=("adamw" "adam")
MAX_TRAIN_STEPS=("30000")

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="accelerate_configs/uncompiled_8.yaml" #Default: "accelerate_configs/uncompiled_1.yaml"

# Absolute path to where the data is located. Make sure to have read the README for how to prepare data.
# This example assumes you downloaded an already prepared dataset from HF CLI as follows:
#   huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir /path/to/my/datasets/disney-dataset
DATA_ROOT="datasets/video-dataset-disney"
DATA_ROOT="datasets/Single-Sample-Disney-VideoGeneration-Dataset"
CAPTION_COLUMN="prompt.txt"
VIDEO_COLUMN="videos.txt"


#(LORA_ALPHA / RANK) = the lora's strength on a scale from 0 to 1
#https://datascience.stackexchange.com/questions/123229/understanding-alpha-parameter-tuning-in-lora-paper
RANK=2048 #Default: 128 
LORA_ALPHA=$RANK

NUM_DATAWORKERS=0 #0 means on main thread. 8 was default.
GRADIENT_ACCUMULATION_STEPS=1 #Default: 1
CHECKPOINTING_STEPS=200 #Default=1000
VALIDATION_EPOCHS=100 #Default: 10, equivalent to 690 here

ryan_data_debug='True'
ryan_data_post_noise_alpha='0,1'
ryan_data_delegator_address='100.118.167.201'
ryan_data_noise_downtemp_interp='blend_norm' #nearest, blend, blend_norm

#Get a unique date string so we can have a unique output folder
export TZ="America/New_York"
DATESTRING=$(date +"%Y-%m-%dT%H-%M-%S%z")

#Notes:
# don't worry about id_token our dataset overrides the prompt generation it doesn't matter

# Launch experiments with different hyperparameters
for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do

        #RYAN: Use a local directory on this computer so it  never halts. But also, periodically sync it.
        # output_dir_local="outputs/models/cogvideox-lora__optimizer_${optimizer}__steps_${steps}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}__${DATESTRING}/"
        output_dir_local="outputs/models/cogx-lora-i2v__degrad=${ryan_data_post_noise_alpha}__downtemp=${ryan_data_noise_downtemp_interp}__lr=${learning_rate}__${DATESTRING}/"
        output_dir="/COGVID_OUTPUTS/${output_dir_local}"
        mkdir -p $output_dir
        mkdir -p $output_dir_local
        # Start the syncing
        (
            while true; do
                sleep 30  # sleep for 30 seconds
                rsync -r "$output_dir" "$output_dir_local"
                echo "rsync: Synced $output_dir to $output_dir_local!"
            done
        ) &
        syncing_pid=$!
        #Kill the syncing process when we exit
        trap "echo 'Killing sync process with PID $syncing_pid'; kill $syncing_pid" EXIT

        cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS training/cogvideox_image_to_video_lora.py \
          --pretrained_model_name_or_path THUDM/CogVideoX-5b-I2V \
          --data_root $DATA_ROOT \
          --caption_column $CAPTION_COLUMN \
          --video_column $VIDEO_COLUMN \
          --id_token BW_STYLE \
          --height_buckets 480 \
          --width_buckets 720 \
          --frame_buckets 49 \
          --dataloader_num_workers $NUM_DATAWORKERS \
          --pin_memory \
          --validation_prompt \"BW_STYLE A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance\" \
          --validation_images \"/root/CleanCode/Github/cogvideox-factory/datasets/firstframe_panda.jpg\" \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_epochs $VALIDATION_EPOCHS \
          --seed 42 \
          --rank $RANK \
          --lora_alpha $LORA_ALPHA \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --max_num_frames 49 \
          --train_batch_size 1 \
          --max_train_steps $steps \
          --checkpointing_steps $CHECKPOINTING_STEPS \
          --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
          --gradient_checkpointing \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 400 \
          --lr_num_cycles 1 \
          --enable_slicing \
          --enable_tiling \
          --noised_image_dropout 0.05 \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --nccl_timeout 1800 \
          --ryan_data_debug $ryan_data_debug \
          --ryan_data_post_noise_alpha $ryan_data_post_noise_alpha \
          --ryan_data_delegator_address $ryan_data_delegator_address \
          --ryan_data_noise_downtemp_interp $ryan_data_noise_downtemp_interp"
        
        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
