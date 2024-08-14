export PREVIOUS_MODEL_DIR="runwayml/stable-diffusion-v1-5"
# export PREVIOUS_MODEL_DIR="saved_models/EXAMPLE"
export MODEL_NAME="$PREVIOUS_MODEL_DIR"
export dataset_name="../data/prepare_data/example_data_train_val_test/data_arrow/dataset_with_text_MERGED"
export new_train_steps=10000
export output_dir="saved_models/EXAMPLE"


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
  --checkpointing_steps=500 \
  --validation_steps=500 \
  --checkpoints_total_limit=2 \
  --report_to="wandb" \
  --snr_gamma=5.0 \
  --noise_offset=0.1 \
  --validation_img_dir="../data/prepare_data/example_data_train_val_test/val" \
  --validation_prompt_dir="../data/prepare_data/example_data_train_val_test/_json" \
  --tracker_project_name="EXAMPLE" \
  --validation_prompts "[1,0,0,0,0,0,0,0,0]" "[0,1,0,0,0,0,0,0,0]" \
  --validate_at_the_end=True \
  --save_at_end=True \
  --use_ema \
  --n_heatmap_images=0 \
  --radius=20 \
  --random_gray_probability=0.0 \
  --hist_shift_prob=0.0 \
  --random_rotation \
  --random_full_rotation \
  --random_flip \
  --min_step_to_validate=0 \
  --random_augment=True \
  --random_augment_validation=False \
  --mixup_alpha=0.3 \
  --color_jitter \
  --jitter_strength=0.25 \
  --n_image_rounds=10 \
  --n_noise_lvls=100 \
  --min_noise_trials=20 \

  # --resume_from_checkpoint="latest" \
