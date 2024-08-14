export previous_train_steps=0
export PREVIOUS_MODEL_DIR="saved_models/${previous_train_steps}-step-model"
export MODEL_NAME="$PREVIOUS_MODEL_DIR"
export dataset_name="../data/data_arrow/data_arrow_anomaly_erythroblast/dataset_with_text_MERGED"
export new_train_steps=0
export total_steps=$((new_train_steps + previous_train_steps))
export output_dir="saved_models/${total_steps}-TEST"

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
  --checkpointing_steps=20000 \
  --validation_steps=20000 \
  --checkpoints_total_limit=2 \
  --report_to="wandb" \
  --snr_gamma=5.0 \
  --noise_offset=0.1 \
  --validation_img_dir="../data/anomaly_erythroblast/test_blast" \
  --validation_prompt_dir="../data/anomaly_erythroblast/test_blast_json" \
  --random_flip \
  --tracker_project_name="anomaly erythroblast" \
  --validate_at_the_end=True \
  --save_at_end=False \
  --random_rotation \
  --random_full_rotation \
  --n_heatmap_images=3 \
  --validation_prompts "[1,0,0,0,0,0,0,0,0]" \
  --use_ema \
  --color_jitter \
  --radius=20 \
  --hist_shift_prob=0.0 \
  --min_step_to_validate=0 \
  --n_image_rounds=5000 \
  --n_noise_lvls=2000 \
  --min_noise_trials=20 \
  --jitter_strength=0.25 \
  --mixup_alpha=0.3 \
  --random_augment=True \
  --random_augment_validation=False \
