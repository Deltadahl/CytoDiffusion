export previous_train_steps=0
export PREVIOUS_MODEL_DIR="saved_models/${previous_train_steps}-step-model"
export MODEL_NAME="$PREVIOUS_MODEL_DIR"
export dataset_name="../data/datasets/PBC_dataset_normal_DIB_train_val_test/data_arrow/dataset_with_text_MERGED"
export new_train_steps=0
export total_steps=$((new_train_steps + previous_train_steps))
export output_dir="saved_models/${total_steps}-TEST"


accelerate launch --mixed_precision="fp16"  diffusion_discriminator.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=360 \
  --train_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=$new_train_steps \
  --learning_rate=1e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=1000 \
  --output_dir="$output_dir" \
  --center_crop \
  --checkpointing_steps=2000 \
  --validation_steps=2000 \
  --checkpoints_total_limit=2 \
  --report_to="wandb" \
  --snr_gamma=5.0 \
  --noise_offset=0.1 \
  --validation_img_dir="../data/datasets/PBC_dataset_normal_DIB_train_val_test/test" \
  --validation_prompt_dir="../data/datasets/PBC_dataset_normal_DIB_train_val_test/_json" \
  --random_flip \
  --tracker_project_name="in-domain PBC" \
  --validate_at_the_end=True \
  --save_at_end=False \
  --random_rotation \
  --random_full_rotation \
  --n_heatmap_images=1 \
  --validation_prompts "[1,0,0,0,0,0,0,0,0]" \
  --radius=20 \
  --hist_shift_prob=0.0 \
  --random_gray_probability=0.0 \
  --use_ema \
  --min_step_to_validate=0 \
  --n_image_rounds=5000 \
  --n_noise_lvls=2000 \
  --min_noise_trials=20 \
  --random_augment=True \
  --random_augment_validation=False \
  --jitter_strength=0.25 \
  --mixup_alpha=0.3 \
  --color_jitter \
