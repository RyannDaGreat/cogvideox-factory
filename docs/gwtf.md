# Go With The Flow (GWTF) Training

This extension enables "Go With The Flow" training, where predefined noise tensors are used during training instead of random noise.

## Usage

### Step 1: Generate Noise Tensors

First, generate noise tensors for your dataset:

```bash
python scripts/gwtf_noise_generator.py \
  --dataset_path /path/to/your/dataset \
  --output_path /path/to/save/noise_tensors \
  --shape 4 16 64 64 \
  --seed 42
```

The shape should match the latent dimensions of your model. For example:
- For Wan I2V models: typically [4, 16, 64, 64] for 512x512 videos
- For CogVideoX models: typically [4, 16, 32, 32] for 256x256 videos

### Step 2: Create a Dataset Config

Create a dataset configuration file `gwtf_dataset_config.json`:

```json
{
  "datasets": [
    {
      "data_root": "/path/to/your/dataset",
      "noise_root": "/path/to/your/noise_tensors",
      "dataset_type": "gwtf", 
      "video_resolution_buckets": [[16, 512, 512]],
      "reshape_mode": "bicubic"
    }
  ]
}
```

### Step 3: Train Your Model

Run training as usual, but use the GWTF dataset:

```bash
python train.py \
  --model_name wan \
  --training_type lora \
  --pretrained_model_name_or_path "Wan-AI/Wan2.1-I2V-1.3B-Diffusers" \
  --dataset_config gwtf_dataset_config.json \
  --rank 64 \
  --lora_alpha 64 \
  --target_modules "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)" \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr 1e-4 \
  --train_steps 1000 \
  --output_dir wan_i2v_gwtf_lora_output
```

This works with any model type (Wan, CogVideoX, etc.) and training type (LoRA, full finetune).

## Technical Details

The implementation works by:

1. Modifying the model specification's forward methods to accept custom noise tensors
2. Using a custom dataset class that loads pre-computed noise tensors alongside videos/images
3. Passing these noise tensors through the data pipeline to the model

The approach is completely compatible with all existing training methods, including LoRA and full finetuning, and works with all supported model types in the codebase:
- Wan
- CogVideoX
- LTX Video
- Flux
- Hunyuan Video
- CogView4