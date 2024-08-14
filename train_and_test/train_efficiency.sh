export previous_train_steps=0
export PREVIOUS_MODEL_DIR="runwayml/stable-diffusion-v1-5"
# export PREVIOUS_MODEL_DIR="saved_models/${previous_train_steps}-step-model"
export MODEL_NAME="$PREVIOUS_MODEL_DIR"
export dataset_name="../data/datasets/PBC_efficiency_train_val_test/data_arrow/dataset_with_text_MERGED"
export new_train_steps=20000
export total_steps=$((new_train_steps + previous_train_steps))
export output_dir="saved_models/${total_steps}-effieicency_10_1"


accelerate launch --mixed_precision="fp16"  diffusion_discriminator.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=360 \
  --train_batch_size=10 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=$new_train_steps \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --output_dir="$output_dir" \
  --center_crop \
  --checkpointing_steps=1000 \
  --validation_steps=1000 \
  --checkpoints_total_limit=2 \
  --report_to="wandb" \
  --snr_gamma=5.0 \
  --noise_offset=0.1 \
  --validation_img_dir="../data/datasets/PBC_efficiency_train_val_test/val" \
  --validation_prompt_dir="../data/datasets/PBC_efficiency_train_val_test/_json" \
  --random_flip \
  --tracker_project_name="efficiency" \
  --validate_at_the_end=True \
  --save_at_end=False \
  --random_rotation \
  --random_full_rotation \
  --n_heatmap_images=1 \
  --validation_prompts "[1,0,0,0,0,0,0,0,0]" \
  --color_jitter \
  --radius=20 \
  --hist_shift_prob=0.0 \
  --random_gray_probability=0.0 \
  --use_ema \
  --min_step_to_validate=0 \
  --n_image_rounds=300 \
  --n_noise_lvls=35 \
  --min_noise_trials=5 \
  --random_augment=True \
  --random_augment_validation=False \
  --mixup_alpha=0.3 \
  --jitter_strength=0.25 \
