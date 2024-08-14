#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Code modified by Simon Deltadahl: scfc3@cam.ac.uk

import ast
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import json
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    PNDMScheduler,
    LMSDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from torchvision.transforms import functional as TF
from PIL import Image
import glob
import re
import matplotlib.pyplot as plt
from torch import autocast
from scipy.stats import ttest_rel
import matplotlib.colors as mcolors
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import io
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import pickle
import scipy.stats as stats
from matplotlib.patches import Patch
from random import Random
from scipy.stats import gaussian_kde, wasserstein_distance
import pandas as pd
import uuid
from torchvision.transforms import RandAugment
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def str_to_float_list(s):
    try:
        # Convert string representation of list into actual list
        values = ast.literal_eval(s)
        if type(values) == list and all(isinstance(item, (float, int)) for item in values):
            return [float(item) for item in values]  # Convert all items to float
        else:
            raise ValueError
    except:
        raise argparse.ArgumentTypeError("Input must be a list of floats or integers")


# def convert_dim(caption, target_shape=(77, 768)):
#     caption = np.array(caption)
#     # Initialize a 2D array of zeros with the desired final shape
#     final_array = np.zeros(target_shape)
#     # Determine the length of the caption to be inserted
#     assert target_shape[1] >= len(caption)
#     caption_len = len(caption)
#     # Insert the caption into the beginning of the final_array for all rows
#     for i in range(target_shape[0]):
#         final_array[i, 0:caption_len] = caption
#     final_array = np.array(final_array)
#     return final_array

def convert_dim(caption, target_shape=(77, 768)):
    caption = np.array(caption)
    final_array = np.zeros(target_shape)
    caption_len = len(caption)

    # Calculate the number of full rows needed
    num_full_rows = caption_len // target_shape[1]

    # Calculate the remaining elements after filling full rows
    remaining_elements = caption_len % target_shape[1]

    # Fill full rows
    for i in range(num_full_rows):
        row_start = i * target_shape[1]
        row_end = row_start + target_shape[1]
        final_array[i, :] = caption[row_start:row_end]

    # Fill the remaining elements in the next row
    if remaining_elements > 0:
        final_array[num_full_rows, :remaining_elements] = caption[num_full_rows * target_shape[1]:]

    # Repeat the filled rows until the target shape is reached
    num_filled_rows = num_full_rows + (1 if remaining_elements > 0 else 0)
    for i in range(num_filled_rows, target_shape[0]):
        final_array[i, :] = final_array[i % num_filled_rows, :]

    return final_array

class RandHistogramShift:
    def __init__(self, hist_shift_prob, num_control_points=10):
        self.prob = hist_shift_prob
        self.num_control_points = num_control_points
        self.rand = Random()

    def randomize(self):
        if self.rand.random() > self.prob:
            return None  # Skip transformation with the probability of 1 - prob
        num_control_points = self.num_control_points
        reference = np.linspace(0, 1, num_control_points)
        floating = np.copy(reference)
        for i in range(1, num_control_points - 1):
            floating[i] = self.rand.uniform(floating[i - 1], floating[i + 1])
        return reference, floating

    def __call__(self, img):
        params = self.randomize()
        if params is None:
            return img  # No transformation is applied
        reference, floating = params
        img = TF.to_tensor(img)  # Convert PIL Image to Tensor
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)  # Normalize to [0, 1]
        img_interp = np.interp(img.numpy(), reference, floating)
        img_interp = torch.tensor(img_interp) * (img_max - img_min) + img_min  # Rescale back
        return TF.to_pil_image(img_interp)  # Convert Tensor back to PIL Image


# This is the most imporant function in this script. It is responsible for running the validation loop.
def log_validation(vae_orig, unet_orig, args, accelerator, weight_dtype, epoch, scheduler):
    logger.info("Running validation... ")
    scheduler.set_timesteps(50)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    vae=accelerator.unwrap_model(vae_orig)
    unet=accelerator.unwrap_model(unet_orig)

    images_val_prompts = []
    # Generate images for the validation prompts to see how well the model can generate images from the prompts
    if args.validation_prompts:
        for k in range(len(args.validation_prompts)):
            latents = torch.randn(
                (1, unet.config.in_channels, args.resolution // 8, args.resolution // 8),
                    generator=generator,
            ).to(accelerator.device, dtype=weight_dtype)
            latents = (latents * scheduler.init_noise_sigma)

            with torch.autocast("cuda"):
                real_input = args.validation_prompts[k]

                real_input_mat = convert_dim(real_input)
                real_input_mat = torch.tensor(real_input_mat)
                real_input_mat = real_input_mat.to(accelerator.device, dtype=weight_dtype)
                input_mat = real_input_mat.unsqueeze(0)

                for j, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
                    latent_model_input = scheduler.scale_model_input(latents, t)
                    latent_model_input = latent_model_input.to(accelerator.device, dtype=weight_dtype)

                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_mat).sample

                    latents = scheduler.step(noise_pred, t, latents).prev_sample

            # scale and decode the image latents with vae
            latents_scaled = 1 / vae.config.scaling_factor * latents
            with torch.no_grad():
                latents_scaled = latents_scaled.to(dtype=weight_dtype)
                image = vae.decode(latents_scaled)[0]

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().float().permute(0, 2, 3, 1).numpy()
            image = (image * 255).round().astype("uint8")
            images_val_prompts.append(image)

    def rotate_image(image, angle):
        """Rotate the image by a given angle."""
        return TF.rotate(image, angle)

    def flip_image_horizontally(image, should_flip):
        """Optionally flip the image horizontally."""
        if should_flip:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def diagonal_mirror_image(image, should_mirror):
        """Optionally apply diagonal mirroring."""
        if should_mirror:
            # Flip horizontally and then transpose
            return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
        return image

    def pil_to_latent_color_jitter(input_im, angle, horizontal_flip, diagonal_mirror):
        """Apply a sequence of transformations to the image."""
        if input_im.mode != 'RGB':
            input_im = input_im.convert('RGB')

        preprocess = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            RandHistogramShift(args.hist_shift_prob),
            transforms.ColorJitter(brightness=args.jitter_strength, contrast=args.jitter_strength,
                                saturation=args.jitter_strength, hue=(args.jitter_strength/2)),
            transforms.RandomGrayscale(p=args.random_gray_probability),
            transforms.Lambda(lambda x: RandAugment()(x)) if args.random_augment_validation else transforms.Lambda(lambda x: x),
            transforms.Lambda(lambda x: flip_image_horizontally(x, horizontal_flip)),
            transforms.Lambda(lambda x: diagonal_mirror_image(x, diagonal_mirror)),
            transforms.ToTensor(),  # Convert to tensor
            transforms.Lambda(lambda x: rotate_image(x, angle)),
            lambda x: x.unsqueeze(0).to(accelerator.device, dtype=weight_dtype) * 2 - 1,
        ])

        # Apply the transformations and encode
        latent = vae.encode(preprocess(input_im))
        return vae.config.scaling_factor * latent.latent_dist.sample()

    # NOTE: Feel free to change this dict if you are using other classes/data
    category_to_number = {
        "basophil": 0,
        "blast": 1,
        "lymphocyte": 2,
        "monocyte": 3,
        "neutrophil": 4,
        "erythroblast": 5,
        "eosinophil": 6,
        "immature granulocyte": 7,
        "platelet": 8,
        "artefact": 9,
    }

    number_to_category = {v: k for k, v in category_to_number.items()}
    all_classification_images = []
    all_heatmaps_images = []
    correct_tracker = []

    image_paths = []
    prompts_all = []
    unique_prompts_short = []
    unique_prompts = []

    assert args.validation_prompt_dir is not None, "Validation prompt directory is not provided"
    classes_to_skip = args.classes_to_skip_for_validation if args.classes_to_skip_for_validation is not None else []

    print(f"args.classes_to_skip_for_validation {args.classes_to_skip_for_validation}")
    # Iterate through the subdirectories and files in the base directory
    for class_name in sorted(os.listdir(args.validation_img_dir)):
        class_dir = os.path.join(args.validation_img_dir, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                if os.path.isfile(image_path):
                    # Add the image path to the list
                    image_paths.append(image_path)

                    class_name_processed = class_name.replace("_", " ")

                    for key, value in category_to_number.items():
                        class_name_processed = class_name_processed.replace(str(value), key)

                    json_filename = re.sub(r'\.png|\.jpeg|\.jpg|\.bmp|\.tiff|\.tif', '.json', image_name, flags=re.IGNORECASE)
                    assert json_filename.endswith('.json'), f"File {json_filename} does not end with .json"

                    json_path = os.path.join(args.validation_prompt_dir, os.path.join(class_name, json_filename))
                    with open(json_path, 'r') as file:
                        prompt = json.load(file)
                        assert isinstance(prompt, list), "Prompt is not a list"

                    prompts_all.append(prompt)
                    if class_name_processed not in unique_prompts_short:
                        if class_name_processed not in classes_to_skip:
                            print(f"Adding {class_name_processed} to unique prompts")
                            unique_prompts.append(prompt)
                            unique_prompts_short.append(class_name_processed)

    if args.unique_prompts_to_add is not None:
        for prompt in args.unique_prompts_to_add:
            if prompt not in unique_prompts:
                unique_prompts.append(prompt)
                unique_prompts_short.append(prompt)


    text_embeddings = torch.tensor(np.array([convert_dim(unique_prompts[i]) for i in range(len(unique_prompts))])).to(accelerator.device, dtype=weight_dtype)

    n_noise_lvls = args.n_noise_lvls
    if args.min_noise_trials is None:
        min_noise_trials = n_noise_lvls
    else:
        min_noise_trials = args.min_noise_trials   # same if we are not pruning away unlikely classes
    n_image_rounds = args.n_image_rounds

    batch_size = 15
    num_inference_steps_classification = 1000
    noise_list = np.linspace(1, num_inference_steps_classification-1, n_noise_lvls, dtype=int)
    random.shuffle(noise_list)

    scheduler.set_timesteps(num_inference_steps_classification)
    n_classes = len(unique_prompts)
    assert n_classes == len(unique_prompts_short), "The number of classes and unique prompts short should be the same"
    correct_counts = np.zeros(n_classes, dtype=int)
    total_counts = np.zeros(n_classes, dtype=int)

    TP = np.zeros(n_classes)
    FP = np.zeros(n_classes)
    FN = np.zeros(n_classes)
    TN = np.zeros(n_classes)
    images_in_test_set = np.ones(n_classes, dtype=int) * n_image_rounds
    conf_matrix = np.zeros((n_classes, n_classes))

    avg_curve_correct = []
    avg_curve_wrong = []
    delta_correct = []
    heatmap_count = 0

    # unique_prompts_short
    all_preds = []
    all_next_most_probable = []
    all_labels = []
    all_errors = []
    all_paths = []
    all_area_under_mean_curve = []
    all_loss_saver = []
    all_t = []

    for image_round in range(n_image_rounds):
        for correct_index, prompt in enumerate(unique_prompts):
            if heatmap_count < args.n_heatmap_images:
                heatmap_error = torch.zeros((n_classes + 1, unet.config.in_channels, args.resolution // 8, args.resolution // 8)).to(accelerator.device, dtype=weight_dtype)
            else:
                heatmap_error = None

            class_images = [f for f, p in zip(image_paths, prompts_all) if p == prompt]
            if args.shuffle_images:
                random.shuffle(class_images)

            # save the timesteps
            if image_round == 0 and correct_index == 0:
                for noise_idx, start_step in enumerate(noise_list):
                    t = scheduler.timesteps[start_step]
                    all_t.append(t.item())

            if len(class_images) <= image_round:
                images_in_test_set[correct_index] = len(class_images)
                continue

            selected_img_path = class_images[image_round]
            print(f"Processing {selected_img_path} from class {prompt}")

            indices_to_update = list(range(n_classes))
            nr_updates_per_index = np.zeros(n_classes, dtype=int)
            mse_latant_lists = np.zeros((n_noise_lvls + 1, n_classes))
            loss_saver = np.zeros((n_noise_lvls, n_classes))
            mse_latent_normalized = np.zeros((n_noise_lvls + 1, n_classes))

            input_image = Image.open(selected_img_path)
            input_image = input_image.resize((args.resolution, args.resolution))

            with tqdm(enumerate(noise_list), total=len(noise_list)) as pbar:
                for noise_idx, start_step in pbar:
                    single_noise = torch.randn((1, unet.config.in_channels, args.resolution // 8, args.resolution // 8), generator=generator).to(accelerator.device)
                    noise = single_noise.repeat(n_classes, 1, 1, 1)  # using the same noise for all prompts

                    diagonal_mirror = False
                    horizontal_flip = False
                    angle = 0
                    if args.random_full_rotation:
                        angle = random.randint(0, 359)
                        diagonal_mirror = random.random() > 0.5
                    elif args.random_rotation:
                        angle = random.choice([0, 90, 180, 270])
                    if args.random_flip:
                        horizontal_flip = random.random() > 0.5

                    encoded = pil_to_latent_color_jitter(input_image, angle, horizontal_flip, diagonal_mirror)

                    latents = scheduler.add_noise(encoded, noise, timesteps=torch.tensor([scheduler.timesteps[start_step]]))
                    latents = latents.to(accelerator.device, dtype=weight_dtype)
                    with autocast("cuda"):
                        noise_pred = torch.zeros((n_classes, unet.config.in_channels, args.resolution // 8, args.resolution // 8), dtype=unet.dtype).to(accelerator.device)

                        t = scheduler.timesteps[start_step]
                        latent_model_input = scheduler.scale_model_input(latents, t)
                        for i in range(0, len(indices_to_update), batch_size):
                            batch_indices = indices_to_update[i:i+batch_size]
                            with torch.no_grad():
                                noise_pred[batch_indices, :, :, :] = unet(
                                    latent_model_input[batch_indices, :, :, :],
                                    t,
                                    encoder_hidden_states=text_embeddings[batch_indices, :, :]
                                ).sample

                    # NOTE: This mask is used to only consider the pixels inside the circle. If you do want to take the loss of the whole image, you can remove this mask or just set the radius to a large number.
                    H, W = args.resolution // 8, args.resolution // 8
                    radius = args.radius
                    # Create a grid of coordinates
                    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                    center_x, center_y = W // 2, H // 2
                    # Calculate the distance of each pixel from the center
                    dist_from_center = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                    # Create the mask: 1 inside the circle, 0 outside
                    mask = (dist_from_center <= radius).float()
                    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 1, 1).to(accelerator.device)
                    mask = mask_expanded.permute(0, 3, 1, 2)

                    itterations = len(indices_to_update)
                    for i in range(itterations):
                        prompt_idx = indices_to_update[i]

                        generated = noise_pred[prompt_idx]
                        original = noise[prompt_idx]
                        masked_generated = generated * mask
                        masked_original = original * mask

                        if heatmap_count < args.n_heatmap_images:
                            def rotate_tensor(tensor, angle):
                                rotated_images = []
                                for image in tensor:
                                    rotated_channels = []
                                    for channel in image:
                                        rotated_channel = TF.rotate(channel.unsqueeze(0), angle, fill=0).squeeze(0)
                                        rotated_channels.append(rotated_channel)
                                    rotated_image = torch.stack(rotated_channels, dim=0)
                                    rotated_images.append(rotated_image)
                                return torch.stack(rotated_images, dim=0)

                            # Rotate back to the original angle so that the heatmap is aligned with the original image
                            masked_generated = rotate_tensor(masked_generated, -angle)
                            masked_original = rotate_tensor(masked_original, -angle)

                            if horizontal_flip:
                                for j in range(4):
                                    # Flip horizontally
                                    masked_generated[0,j,:,:] = torch.flip(masked_generated[0,j,:,:], [1])
                                    masked_original[0,j,:,:] = torch.flip(masked_original[0,j,:,:], [1])
                            if diagonal_mirror:
                                for j in range(4):
                                    # diagonal mirror
                                    masked_generated[0,j,:,:] = torch.rot90(torch.flip(masked_generated[0,j,:,:], [1]), 1, [0, 1])
                                    masked_original[0,j,:,:] = torch.rot90(torch.flip(masked_original[0,j,:,:], [1]), 1, [0, 1])

                            heatmap_error[prompt_idx,:,:,:] += (masked_original[0,:,:,:] - masked_generated[0,:,:,:]) / n_noise_lvls

                        normalise = False  # Not used at the moment, but this can be used to normalise the mse values to N(0,1)
                        if args.use_snr_weighting:
                            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                            # This is discussed in Section 4.2 of the same paper.

                            snr = compute_snr(scheduler, t)
                            mse_loss_weights = torch.min(snr, args.snr_gamma * torch.ones_like(torch.ones(1))).to(accelerator.device, dtype=weight_dtype)
                            if scheduler.config.prediction_type == "epsilon":
                                mse_loss_weights = mse_loss_weights / snr
                            elif scheduler.config.prediction_type == "v_prediction":
                                mse_loss_weights = mse_loss_weights / (snr + 1)

                            loss = F.mse_loss(masked_original, masked_generated, reduction="sum")
                            loss_saver[noise_idx, prompt_idx] += loss.item()
                            mse_latant_lists[noise_idx, prompt_idx] += loss.item() * mse_loss_weights
                        else:
                            loss = F.mse_loss(masked_original, masked_generated, reduction="sum")
                            loss_saver[noise_idx, prompt_idx] += loss.item()

                            def poly4(t):
                                a_arr = args.poly_coeff
                                if a_arr is None:
                                    a_arr_new =  [0.9270553940204841, 9.836536887830196, -4.6840608002246205, 6.018538389994896, -9.339786177681749]
                                    a0, a1, a2, a3, a4 = a_arr_new
                                    return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4

                                a_arr = a_arr[0]
                                assert len(a_arr) == 5, f"The polynomial should have 5 coefficients {a_arr}"
                                a0, a1, a2, a3, a4 = a_arr
                                return a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4

                            def custom_weighting_poly4():
                                t_tmp = t.item()
                                t_tmp = (1000 - t_tmp) / 1000

                                poly_value = poly4(t_tmp)
                                epsilon = 1e-10
                                weight = 1 / (poly_value + epsilon)

                                assert weight >= 0, f"Weight is negative: {weight}"
                                return max(weight,0)

                            weight = custom_weighting_poly4()
                            mse_latant_lists[noise_idx, prompt_idx] += loss.item() * weight

                    if normalise:
                        row_mean = np.mean(mse_latant_lists[noise_idx, indices_to_update])
                        row_std = np.std(mse_latant_lists[noise_idx, indices_to_update])
                        if row_std > 0:  # To avoid division by zero
                            mse_latent_normalized[noise_idx, indices_to_update] = (mse_latant_lists[noise_idx, indices_to_update] - row_mean) / row_std
                        else:
                            # Handle the case where the std is 0; it means all values in the row are the same.
                            mse_latent_normalized[noise_idx, indices_to_update] = mse_latant_lists[noise_idx, indices_to_update] - row_mean

                    # Subtract the minimum value from all elements in the row
                    row_min = np.min(mse_latant_lists[noise_idx, indices_to_update])
                    mse_latent_normalized[noise_idx, indices_to_update] = mse_latant_lists[noise_idx, indices_to_update] - row_min

                    # for the values not in indices_to_update, set them to the mean of all previous rows
                    for i in range(n_classes):
                        if i not in indices_to_update:
                            mse_latent_normalized[noise_idx, i] = mse_latent_normalized[:noise_idx, i].sum() / noise_idx

                    nr_updates_per_index[indices_to_update] += 1
                    if not normalise:
                        mask = mse_latent_normalized > 0
                        mask[-1, :] = 0
                        denominator = 1
                        for i in indices_to_update:
                            mse_latent_normalized[-1, i] = mse_latent_normalized[:-1, i].sum() / (denominator * nr_updates_per_index[i])

                    mse_values_formatted = [f"{val:.2f}" for val in mse_latent_normalized[-1, indices_to_update]]
                    pbar.set_postfix({"mse": mse_values_formatted}, refresh=True)

                    # Sort indices based on mse values and then by index to ensure determinism
                    sorted_indices = sorted(range(len(mse_latent_normalized[-1, :])),
                                            key=lambda x: (mse_latent_normalized[-1, x], x))

                    # Using the sorted indices to get the sorted values
                    sorted_values = [mse_latent_normalized[-1, i] for i in sorted_indices]

                    # Create a mapping from original indices to new sorted indices
                    index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(sorted_indices)}

                    # Identifying the most probable class index and the next most probable class index
                    for idx in sorted_indices:
                        if idx in indices_to_update:
                            most_probable_class_idx = idx
                            for idx_2 in sorted_indices:
                                if idx_2 in indices_to_update and idx_2 != idx:
                                    next_most_probable_class_idx = idx_2
                                    break
                            break

                    # Pruning:
                    if noise_idx >= min_noise_trials and noise_idx % 5 == 0:
                        indices_to_remove = []
                        for class_idx in indices_to_update:
                            if class_idx == most_probable_class_idx:
                                continue
                            # Perform a dependent t-test comparing each class to the most probable class
                            stat, p_value = ttest_rel(mse_latent_normalized[:noise_idx + 1, class_idx],
                                                    mse_latent_normalized[:noise_idx + 1, most_probable_class_idx])

                            if p_value < 2e-3:
                                # remove the class if it significantly differs (and performs worse) than the most probable class
                                if mse_latent_normalized[-1, class_idx] > mse_latent_normalized[-1, most_probable_class_idx]:
                                    indices_to_remove.append(class_idx)

                        # Remove all indices that meet the criteria
                        for idx in indices_to_remove:
                            indices_to_update.remove(idx)

                        if len(indices_to_update) <= 1:
                            break

            if heatmap_count < args.n_heatmap_images:
                heatmap_count += 1
                assert heatmap_error.shape[1] == 4, "The heatmap should have 4 channels"
                heatmap_error = heatmap_error - heatmap_error[most_probable_class_idx]

                fig, axs = plt.subplots(n_classes, 4, figsize=(30, 9 * (n_classes)))

                for prompt_idx in range(n_classes):
                    fontsize = 18
                    latents = heatmap_error[prompt_idx].unsqueeze(0)
                    latents = (1 / vae.config.scaling_factor) * latents * 28  # Arbitrary scaling factor, adjust as needed

                    with torch.no_grad():
                        latents = latents.to(dtype=weight_dtype)
                        image = vae.decode(latents).sample

                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.detach().cpu().float().permute(0, 2, 3, 1).numpy()
                    images = (image * 255).round().astype("uint8")
                    pil_images = [Image.fromarray(image) for image in images]
                    image = pil_images[0]

                    def rgb2gray(rgb):
                        return np.dot(rgb[..., :3], [1/3, 1/3, 1/3])

                    # Normalize the image
                    def normalize_fixed_range(gray_image, min_val=0, max_val=255):
                        return np.clip((gray_image - min_val) / (max_val - min_val), 0, 1)

                    image_np = np.array(image)
                    gray_image = rgb2gray(image_np)
                    gray_image_normalized = normalize_fixed_range(gray_image)
                    cmap_colors = [(0.267004, 0.004874, 0.329415, 1.0), (0.190631, 0.407061, 0.556089, 1.0), (0.20803, 0.718701, 0.472873, 1.0), (0.993248, 0.906157, 0.143936, 1.0)]

                    from matplotlib.colors import LinearSegmentedColormap
                    custom_cmap = LinearSegmentedColormap.from_list("blue_to_white", cmap_colors, N=256)
                    viridis_image = custom_cmap(gray_image_normalized)[:, :, :3]  # Take only RGB, ignore alpha
                    channel_stats = [
                        {'mean': 0.16519577 , 'std': 0.0337721},
                        {'mean': 0.49670708, 'std': 0.0639124},
                        {'mean': 0.54763337, 'std': 0.02108916}
                    ]

                    k = 3.5  # Threshold factor, arbitrary value, adjust as needed
                    masks = []
                    for i, statsistics in enumerate(channel_stats):
                        lower_thresh = statsistics['mean'] - k * statsistics['std']
                        upper_thresh = statsistics['mean'] + k * statsistics['std']
                        # Create mask for current channel
                        channel_mask = (viridis_image[:,:,i] > lower_thresh) & (viridis_image[:,:,i] < upper_thresh)
                        masks.append(channel_mask)

                    # Combine masks for all channels (logical AND across all channel masks)
                    final_mask = np.logical_and.reduce(masks)

                    # check if input_image is rgb, with 3 channels, otherwise print path to image
                    if input_image.mode != 'RGB':
                        print(f"Image {selected_img_path} is not RGB, but {input_image.mode}")
                        input_image = input_image.convert('RGB')

                    # Overlay heatmap on the real image
                    input_image_np = np.array(input_image)  # Convert tensor to numpy if not already
                    lambda_ = 0.0
                    viridis_scaled = viridis_image * 255

                    # Create the mixed image
                    mixed_image = (lambda_ * input_image_np + (1 - lambda_) * viridis_scaled).astype('uint8')

                    # Initialize the overlay image with the mixed image
                    overlay = np.copy(mixed_image)

                    # Apply mask: where the mask is True, replace with input_image_np; otherwise keep the mixed image
                    overlay[final_mask] = input_image_np[final_mask]

                    # Plot original, heatmap, and overlay in the respective subplots
                    axs[prompt_idx, 0].imshow(input_image)
                    axs[prompt_idx, 0].axis('off')
                    axs[prompt_idx, 0].set_title(f'Original Image, class: "{unique_prompts_short[correct_index]}"', fontsize=fontsize)

                    axs[prompt_idx, 1].imshow(viridis_image)
                    axs[prompt_idx, 1].axis('off')
                    axs[prompt_idx, 1].set_title(f'Heatmap for prompt "{unique_prompts_short[prompt_idx]}"', fontsize=fontsize)

                    # Calculate the mean heatmap error image across the channels
                    mean_heatmap_error = torch.mean(heatmap_error[prompt_idx], dim=0)

                    axs[prompt_idx, 3].imshow(mean_heatmap_error.cpu().float().numpy(), cmap='viridis')
                    axs[prompt_idx, 3].axis('off')
                    axs[prompt_idx, 3].set_title('Mean Heatmap in latent space', fontsize=fontsize)

                    axs[prompt_idx, 2].imshow(overlay)
                    axs[prompt_idx, 2].axis('off')
                    axs[prompt_idx, 2].set_title('Overlay of Heatmap on Original', fontsize=fontsize)

                    if args.save_heatmaps_locally:
                        # Save original image
                        original_dir_path = f'heatmaps/original/correct_{unique_prompts_short[correct_index]}'
                        os.makedirs(original_dir_path, exist_ok=True)

                        # Save heatmap as RGB image
                        heatmap_dir_path = f'heatmaps/rgb/correct_{unique_prompts_short[correct_index]}/conditioned_{unique_prompts_short[prompt_idx]}'
                        heatmap_dir_path_raw = f'heatmaps/rgb_raw/correct_{unique_prompts_short[correct_index]}/conditioned_{unique_prompts_short[prompt_idx]}'
                        heatmap_dir_path_matlab = f'heatmaps/matlab/correct_{unique_prompts_short[correct_index]}/conditioned_{unique_prompts_short[prompt_idx]}'
                        heatmap_dir_path_matlab_raw = f'heatmaps/matlab_raw/correct_{unique_prompts_short[correct_index]}/conditioned_{unique_prompts_short[prompt_idx]}'
                        os.makedirs(heatmap_dir_path, exist_ok=True)
                        os.makedirs(heatmap_dir_path_raw, exist_ok=True)
                        os.makedirs(heatmap_dir_path_matlab, exist_ok=True)
                        os.makedirs(heatmap_dir_path_matlab_raw, exist_ok=True)

                        if prompt_idx == 0:
                            original_file_path = os.path.join(original_dir_path, f'{image_round}.png')
                            input_image_np = np.array(input_image)  # Convert tensor to numpy if not already
                            Image.fromarray(input_image_np).save(original_file_path)
                            print(f"Original image saved to {original_file_path}")

                        heatmap_rgb_path = os.path.join(heatmap_dir_path, f'{image_round}.png')
                        plt.imsave(heatmap_rgb_path, viridis_image, cmap=custom_cmap)
                        heatmap_rgb_path_raw = os.path.join(heatmap_dir_path_raw, f'{image_round}.png')
                        Image.fromarray(image_np).save(heatmap_rgb_path_raw)
                        print(f"Heatmap RGB image saved to {heatmap_rgb_path}")

                        # Change from 0-1 to 0-255
                        gray_image_normalized_255 = (gray_image_normalized * 255).astype(np.uint8)
                        heatmap_mat_path = os.path.join(heatmap_dir_path_matlab, f'{image_round}.mat')
                        savemat(heatmap_mat_path, {'heatmap': gray_image_normalized_255})
                        heatmap_mat_path_raw = os.path.join(heatmap_dir_path_matlab_raw, f'{image_round}.mat')
                        savemat(heatmap_mat_path_raw, {'heatmap': image_np})
                        print(f"Heatmap MATLAB file saved to {heatmap_mat_path}")

                fig.suptitle(f'Heatmap analysis, true class: "{unique_prompts_short[correct_index]}", predicted class "{unique_prompts_short[most_probable_class_idx]}"', fontsize=fontsize * 2)

                # Save to wandb
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                heatmap_image = Image.open(buf)
                all_heatmaps_images.append(heatmap_image)

            predicted_class = most_probable_class_idx
            if predicted_class == correct_index:
                correct_counts[correct_index] += 1
                delta_correct_nr = 1
                TP[correct_index] += 1
            else:
                delta_correct_nr = 0
                FP[predicted_class] += 1
                FN[correct_index] += 1
            for j in range(n_classes):
                if j != correct_index and j != predicted_class:
                    TN[j] += 1


            if not args.use_snr_weighting:
                mse_latent_normalized

            current_delta_correct = mse_latent_normalized[-1, next_most_probable_class_idx] - mse_latent_normalized[-1, predicted_class]
            delta_correct.append((current_delta_correct, delta_correct_nr))

            conf_matrix[correct_index, predicted_class] += 1

            all_preds.append(unique_prompts_short[predicted_class])
            all_next_most_probable.append(unique_prompts_short[next_most_probable_class_idx])
            all_labels.append(unique_prompts_short[correct_index])
            all_errors.append(current_delta_correct)
            all_paths.append(selected_img_path)
            all_loss_saver.append(loss_saver)

            def calculate_shifted_area(sorted_data):
                # Calculate the shift needed to make the smallest value zero
                shift = abs(sorted_data[0])

                # Shift all values
                shifted_data = sorted_data + shift
                assert np.all(np.diff(shifted_data) >= 0), "Data is not sorted"
                assert shifted_data[0] > -1e-6, "Smallest value is not > -1e-6"
                assert shifted_data[-1] < 5000, "Largest value is not < 5000"

                # Calculate the area under the shifted curve
                average_area = np.trapz(shifted_data, dx=1) / len(shifted_data)

                return average_area

            area_under_curve = calculate_shifted_area(mse_latent_normalized[-1, sorted_indices])
            all_area_under_mean_curve.append(area_under_curve)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

            sorted_mse_latent_normalized = mse_latent_normalized[:, sorted_indices]
            correct_value_latent = mse_latent_normalized[-1][correct_index]
            predicted_value_latent = mse_latent_normalized[-1, predicted_class]

            if correct_index == predicted_class:
                avg_curve_correct.append(sorted_mse_latent_normalized[-1, :])
            else:
                avg_curve_wrong.append(sorted_mse_latent_normalized[-1, :])


            cmap = plt.get_cmap('viridis')
            # Normalize the all_t values to the range [0, 1]
            norm = mcolors.Normalize(vmin=np.min(all_t), vmax=np.max(all_t))

            for i in range(len(sorted_mse_latent_normalized[:-1,0])):
                color = cmap(norm(all_t[i]))
                ax1.plot(sorted_mse_latent_normalized[i,:], label='_nolegend_', marker='o', color=color, linewidth=1, linestyle='-', markersize=1, alpha=1)

            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax1)
            cbar.set_label('all_t values')

            ax1.plot(sorted_mse_latent_normalized[-1, :], label='Mean MSE', marker='o', linestyle='-', linewidth=3, color='black')
            highlight_style = dict(marker='*', markersize=15, markeredgecolor='black', markerfacecolor='yellow', zorder=5)

            ax1.plot(index_map[correct_index], correct_value_latent, **highlight_style, label='Correct Value')
            # highlight the predicted class
            ax1.axhline(y=predicted_value_latent, color='red', linestyle='-', linewidth=2)
            highlight_style = dict(marker='*', markersize=10, markeredgecolor='blue', markerfacecolor='green', zorder=5)
            ax1.plot(0, predicted_value_latent, **highlight_style, label='Predicted Value')

            x_labels = [unique_prompts_short[i] for i in sorted_indices]
            ax1.set_xticks(range(len(x_labels)))
            ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=20)

            ax1.set_ylabel('Normalised error', fontsize=20)
            ax1.legend()
            ax1.set_title('MSE Latent Loss')

            ax2.imshow(input_image)
            ax2.axis('off')
            ax2.set_title(f'{unique_prompts_short[correct_index]}', fontsize=20)

            plt.subplots_adjust(hspace=0.3)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)

            # Open the image directly from the BytesIO object
            classification_image = Image.open(buf)
            all_classification_images.append(classification_image)
            correct_tracker.append(correct_index==predicted_class)

            total_counts[correct_index] += 1

            if args.save_all_wrong_images and predicted_class != correct_index:
                # save all the wrong images to a folder
                save_path = "wrong_images_1"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                image_name_tmp = selected_img_path.split("/")[-1]
                classification_image.save(os.path.join(save_path, f"{image_name_tmp}.png"))
                print(f"Saved wrong image to {save_path}/{image_name_tmp}.png")

            if total_counts[correct_index] > 0:
                print(f"Class {unique_prompts_short[correct_index]}: Correct {correct_counts[correct_index]}/{total_counts[correct_index]}")
                print(f"Correct {np.sum(correct_counts)}/{np.sum(total_counts)}")
                print(f"Total: Correct % {(100 * np.sum(correct_counts)/np.sum(total_counts)):.2f}")


    ####################################   Plot and log to wandb below   ########################################
    # Initialize dictionaries for storing curves and logging
    logs = {}
    avg_curves = {}
    std_curves = {}

    def log_array_as_artifact(array, name):
        # Convert array to DataFrame
        df = pd.DataFrame(array, columns=[name])
        unique_id = uuid.uuid4().hex

        # Save to CSV with unique filename
        csv_path = f"{name}_{unique_id}.csv"
        df.to_csv(csv_path, index=False)

        # Create an artifact with the unique filename
        artifact = wandb.Artifact(f"{name}_{unique_id}", type='dataset')
        artifact.add_file(csv_path)
        wandb.log_artifact(artifact)
        os.remove(csv_path)
    # Log arrays as artifacts with unique filenames
    log_array_as_artifact(all_preds, "all_preds")
    log_array_as_artifact(all_next_most_probable, "all_next_most_probable")
    log_array_as_artifact(all_labels, "all_labels")
    log_array_as_artifact(all_errors, "all_errors")
    log_array_as_artifact(all_paths, "all_paths")
    log_array_as_artifact(all_area_under_mean_curve, "all_area_under_mean_curve")
    log_array_as_artifact(all_t, "all_t")

    def log_matrix_as_artifact(matrix, name):
        unique_id = uuid.uuid4().hex

        # Convert numpy array to list for JSON serialization
        matrix_list = matrix.tolist()

        # Save to JSON with unique filename
        json_path = f"{name}_{unique_id}.json"
        with open(json_path, 'w') as f:
            json.dump(matrix_list, f)

        # Create an artifact with the unique filename
        artifact = wandb.Artifact(f"{name}_{unique_id}", type='dataset')
        artifact.add_file(json_path)
        wandb.log_artifact(artifact)
        os.remove(json_path)

    all_loss_saver_array = np.array(all_loss_saver)
    log_matrix_as_artifact(all_loss_saver_array, "all_loss_saver")

    # Extracting deltas and correct labels
    deltas = np.array([x[0] for x in delta_correct]).reshape(-1, 1)
    correct = np.array([x[1] for x in delta_correct])

    # Logistic Regression Model
    if len(deltas) > 1 and 0 in correct and 1 in correct:
        lr_model = LogisticRegression()
        lr_model.fit(deltas, correct)

        # Predicting probabilities for the logistic curve
        x_test = np.linspace(min(deltas), max(deltas), 3000).reshape(-1, 1)
        probabilities = lr_model.predict_proba(x_test)[:, 1]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(deltas, correct, color='blue', label='Data Points', alpha=0.08)
        plt.plot(x_test, probabilities, color='red', label='Logistic Regression Fit')

        # Horizontal line at y=
        confidence_level = 0.9
        plt.axhline(y=confidence_level, color='green', linestyle='--', label=f'Confidence Level {confidence_level}')

        # Find intersection
        # Note: The intersection finds where the curve reaches confidence_level probability.
        intersection = x_test[np.abs(probabilities - confidence_level).argmin()]
        plt.axvline(x=intersection, color='purple', linestyle='--', label='Intersection at x={:.2f}'.format(intersection[0]))

        plt.title('Logistic Regression Fit to Data')
        plt.xlabel('Normalised error difference between predicted and next most probable class')
        plt.ylabel('Correctly classified')
        plt.legend(loc='best', fancybox=True, framealpha=0.5)
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)  # Ensure the buffer's read pointer is at the start
        image = np.array(Image.open(buf))
        logs["Logistic Regression"] = wandb.Image(image)

        # Splitting data based on correctness
        correct_deltas = deltas[correct == 1]
        incorrect_deltas = deltas[correct == 0]

        # Fit Gaussian to correct and incorrect data
        mu_correct, std_correct = stats.norm.fit(correct_deltas)
        mu_incorrect, std_incorrect = stats.norm.fit(incorrect_deltas)

        # Create an array of sorted deltas for plotting
        x = np.linspace(min(deltas), max(deltas), 1000)
        x = x.flatten()
        pdf_correct = stats.norm.pdf(x, mu_correct, std_correct)
        pdf_incorrect = stats.norm.pdf(x, mu_incorrect, std_incorrect)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x, pdf_correct, color='blue', label='Correct Classification')
        plt.plot(x, pdf_incorrect, color='red', label='Incorrect Classification')
        plt.fill_between(x, pdf_correct, color='blue', alpha=0.3)
        plt.fill_between(x, pdf_incorrect, color='red', alpha=0.3)
        plt.title('Gaussian Fits for Classification Correctness')
        plt.xlabel('Differance between prediction and next value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)

        correct_deltas = correct_deltas.flatten()
        incorrect_deltas = incorrect_deltas.flatten()

        if correct_deltas.size > 1 and incorrect_deltas.size > 1:
            kde_correct = gaussian_kde(correct_deltas)
            kde_incorrect = gaussian_kde(incorrect_deltas)

            pdf_kde_correct = kde_correct(x)
            pdf_kde_incorrect = kde_incorrect(x)

            # wasserstein_dist = wasserstein_distance(correct_deltas, incorrect_deltas)
            labels = np.concatenate((np.ones_like(correct_deltas), np.zeros_like(incorrect_deltas)))
            scores = np.concatenate((correct_deltas, incorrect_deltas))
            auc = roc_auc_score(labels, scores)

            plt.figure(figsize=(10, 6))
            plt.plot(x, pdf_kde_correct, color='blue', label='Correct Classification')
            plt.plot(x, pdf_kde_incorrect, color='red', label='Incorrect Classification')
            plt.fill_between(x, pdf_kde_correct, color='blue', alpha=0.3)
            plt.fill_between(x, pdf_kde_incorrect, color='red', alpha=0.3)
            plt.title('Kernel Density Estimate for Classification Correctness')
            plt.xlabel('Increasing certainty')
            plt.ylabel('Probability Density')
            plt.grid(True)

            # Add Wasserstein distance to the legend
            plt.plot([], [], ' ', label=f'AUC: {auc:.4f}')
            plt.legend()

            # Save plot to buffer and then to wandb
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = np.array(Image.open(buf))
            plt.close()

            logs["KDE Plot"] = wandb.Image(image)

        correct_area = np.array(all_area_under_mean_curve)[correct == 1]
        incorrect_area = np.array(all_area_under_mean_curve)[correct == 0]

        if correct_area.size > 1 and incorrect_area.size > 1:
            x = np.linspace(min(all_area_under_mean_curve), max(all_area_under_mean_curve), 1000)
            x = x.flatten()
            kde_correct_area = gaussian_kde(correct_area)
            kde_incorrect_area = gaussian_kde(incorrect_area)

            pdf_kde_correct_area = kde_correct_area(x)
            pdf_kde_incorrect_area = kde_incorrect_area(x)

            # wasserstein_dist_area = wasserstein_distance(correct_area, incorrect_area)
            labels = np.concatenate((np.ones_like(correct_area), np.zeros_like(incorrect_area)))
            scores = np.concatenate((correct_area, incorrect_area))
            auc = roc_auc_score(labels, scores)

            plt.figure(figsize=(10, 6))
            plt.plot(x, pdf_kde_correct_area, color='blue', label='Correct Classification')
            plt.plot(x, pdf_kde_incorrect_area, color='red', label='Incorrect Classification')
            plt.fill_between(x, pdf_kde_correct_area, color='blue', alpha=0.3)
            plt.fill_between(x, pdf_kde_incorrect_area, color='red', alpha=0.3)
            plt.title('Kernel Density Estimate for Classification Correctness based on Area')
            plt.xlabel('Increasing certainty')
            plt.ylabel('Probability Density')
            plt.grid(True)

            # Add Wasserstein distance to the legend
            plt.plot([], [], ' ', label=f'AUC: {auc:.4f}')
            plt.legend()

            # Save plot to buffer and then to wandb
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = np.array(Image.open(buf))
            plt.close()

            logs["KDE Area Plot"] = wandb.Image(image)


    # Definitions for plot categories
    titles = ["Correct", "Wrong"]
    curves = [avg_curve_correct, avg_curve_wrong]

    # Calculate average curves and standard deviations
    for curve, title in zip(curves, titles):
        if curve:  # Check if the list is not empty
            data = np.array(curve)
            avg_curve = np.mean(data, axis=0)
            std_curve = np.std(data, axis=0)
            avg_curves[title] = avg_curve
            std_curves[title] = std_curve

    def plot_curves(curve_titles, fig_title, colors=['blue', 'red']):
        fig, ax = plt.subplots(figsize=(10, 6))
        line_styles = ['-', '--']
        legend_handles = []

        for title, color, style in zip(curve_titles, colors, line_styles):
            if title in avg_curves:
                avg_curve = avg_curves[title]
                std_curve = std_curves[title]
                x = np.arange(len(avg_curve))
                line, = ax.plot(x, avg_curve, label=title, linestyle=style, color=color, linewidth=2)
                ax.fill_between(x, avg_curve - std_curve, avg_curve + std_curve, color=color, alpha=0.2)
                legend_handles.append(line)

        # Add a legend handle for the shaded area representing ±1 SD
        sd_patch = Patch(color='gray', alpha=0.2, label='±1 SD')
        legend_handles.append(sd_patch)

        # ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('Classes sorted by error', fontsize=12)
        ax.set_ylabel('Normalised error', fontsize=12)
        ax.set_title(fig_title, fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left', fontsize=10)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = np.array(Image.open(buf))
        return image

    def plot_curves_diff(curve_titles, fig_title, colors=['blue', 'red']):
        fig, ax = plt.subplots(figsize=(10, 6))
        line_styles = ['-', '--']
        legend_handles = []

        for title, color, style in zip(curve_titles, colors, line_styles):
            if title in avg_curves:
                avg_curve = np.diff(avg_curves[title])  # Calculate difference between consecutive points
                var_curve = std_curves[title][:-1]**2 + std_curves[title][1:]**2  # Sum of variances of consecutive points
                std_curve = np.sqrt(var_curve)  # Standard deviation of the difference
                x = np.arange(len(avg_curve))  # Adjust x to match the new length of avg_curve
                line, = ax.plot(x, avg_curve, label=title, linestyle=style, color=color, linewidth=2)
                ax.fill_between(x, avg_curve - std_curve, avg_curve + std_curve, color=color, alpha=0.2)
                legend_handles.append(line)

        # Add a legend handle for the shaded area representing ±1 SD
        sd_patch = Patch(color='gray', alpha=0.2, label='±1 SD')
        legend_handles.append(sd_patch)

        ax.set_ylim(auto=True)  # Automatically adjust based on data range
        ax.set_xlabel('Classes sorted by error', fontsize=12)
        ax.set_ylabel('Difference in Normalised error', fontsize=12)
        ax.set_title(fig_title, fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left', fontsize=10)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = np.array(Image.open(buf))
        return image

    # Generate plots and log them
    corr_wrong = ["Correct", "Wrong"]
    if all(title in avg_curves for title in corr_wrong):
        image_cw = plot_curves(corr_wrong, "Comparison of Correct vs Wrong")
        logs["correct_vs_wrong"] = wandb.Image(image_cw)
        image_cw = plot_curves_diff(corr_wrong, "Comparison of Correct vs Wrong")
        logs["correct_vs_wrong_diff"] = wandb.Image(image_cw)

    for i in range(n_classes):
        print(f"Class {unique_prompts_short[i]}: Correct {correct_counts[i]}/{total_counts[i]}")
    if np.sum(total_counts) > 0:
        print(f"Total: Correct {np.sum(correct_counts)}/{np.sum(total_counts)}")

    for i in range(n_classes):
        print(f"Class {unique_prompts_short[i]}: Correct % {100 * correct_counts[i]/total_counts[i]}")
    if np.sum(total_counts) > 0:
        print(f"Total: Correct % {100 * np.sum(correct_counts)/np.sum(total_counts)}")
    print("Done")

    class_accuracy_logs = {}
    for i in range(n_classes):
        class_accuracy = 100 * correct_counts[i] / total_counts[i] if total_counts[i] > 0 else 0
        print(f"Class {unique_prompts_short[i]}: Correct {correct_counts[i]}/{total_counts[i]} ({class_accuracy:.2f}%)")
        class_accuracy_logs[f"class_{unique_prompts_short[i]}_accuracy"] = class_accuracy

    total_correct = np.sum(correct_counts)
    total_counts_sum = np.sum(total_counts)

    if total_counts_sum > 0:
        total_accuracy = 100 * total_correct / total_counts_sum if total_counts_sum > 0 else 0
        print(f"Total: Correct {total_correct}/{total_counts_sum} ({total_accuracy:.2f}%)")
    else:
        total_accuracy = 0
    class_accuracy_logs["total_accuracy"] = total_accuracy

    precision = np.zeros(n_classes)
    sensitivity = np.zeros(n_classes)
    accuracy = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)

    for class_index in range(n_classes):
        tp = TP[class_index]
        fp = FP[class_index]
        fn = FN[class_index]
        tn = TN[class_index]

        # Calculate precision and sensitivity for each class
        precision[class_index] = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensitivity[class_index] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1-score using the precision and sensitivity
        if precision[class_index] + sensitivity[class_index] > 0:
            f1_score[class_index] = 2 * (precision[class_index] * sensitivity[class_index]) / (precision[class_index] + sensitivity[class_index])
        else:
            f1_score[class_index] = 0

        # Print the metrics for each class
        print(f"Class {unique_prompts_short[class_index]}:")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"Precision: {precision[class_index]:.2f}, Sensitivity (Recall): {sensitivity[class_index]:.2f}")
        print("-" * 50)

    class_accuracy_logs["F1 Score"] = np.mean(f1_score)
    class_accuracy_logs["F1 Score Weighted"] = np.average(f1_score, weights=total_counts)

    # Plot the precision and sensitivity by class
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = [f'{unique_prompts_short[i]} ({images_in_test_set[i]})' for i in range(n_classes)]
    x = np.arange(len(classes))
    width = 0.35

    rects1 = ax.bar(x - width/2, precision, width, label='Precision')
    rects2 = ax.bar(x + width/2, sensitivity, width, label='Sensitivity (Recall)')

    ax.set_ylabel('Scores')
    ax.set_title('Precision and Sensitivity by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')

    # Place the legend outside the plot area
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Update logs with class_accuracy_logs without overwriting existing keys
    for key, value in class_accuracy_logs.items():
        if key not in logs:
            logs[key] = value

    # Add the precision and sensitivity plot
    buf.seek(0)  # Ensure the buffer's read pointer is at the start
    image = np.array(Image.open(buf))
    logs["Precision and Sensitivity Plot"] = wandb.Image(image, caption="Precision and Sensitivity by Class")

    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust the figure size as needed
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Scale')  # Optional: Add label to the color bar

    # Set ticks and labels with adjustments for visibility
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
        yticks=np.arange(conf_matrix.shape[0]),
        xticklabels=unique_prompts_short,
        yticklabels=unique_prompts_short,
        title='Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.15, right=0.95)  # Tweak these values as necessary

    # Loop over data dimensions and create text annotations.
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(int(conf_matrix[i, j]), 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    # Save to BytesIO object for logging
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = np.array(Image.open(buf))
    logs["Confusion Matrix"] = wandb.Image(image)

    # Convert confusion matrix to wandb.Table
    n_classes = len(unique_prompts_short)
    columns = [f"Pred {unique_prompts_short[i]}" for i in range(n_classes)]
    rows = [f"True {unique_prompts_short[i]}" for i in range(n_classes)]
    table = wandb.Table(columns=[""] + columns)  # Adding an empty string for the first column header

    for i, row_label in enumerate(rows):
        table.add_data(row_label, *(conf_matrix[i].tolist()))  # Add each row of the confusion matrix
    wandb.log({"Confusion Matrix Raw": table})
    plt.close()

    def create_loss_dataframe(loss_array, all_t, all_labels):

        loss_array = loss_array[:,:min_noise_trials,:]
        all_t = all_t[:min_noise_trials]

        # Create a list to store all rows
        rows = []

        # Iterate through each image, timestep, and class
        for image_idx in range(loss_array.shape[0]):
            image_class = all_labels[image_idx]
            for timestep_idx in range(loss_array.shape[1]):
                for class_idx in range(loss_array.shape[2]):
                    rows.append({
                        'image_id': image_idx,
                        'image_class': image_class,
                        'timestep': 1000 - all_t[timestep_idx],
                        'test_class': all_labels[class_idx],
                        'loss': loss_array[image_idx, timestep_idx, class_idx]
                    })

        # Create the DataFrame
        df = pd.DataFrame(rows)

        # Convert 'image_id' and 'timestep' to integers
        df['image_id'] = df['image_id'].astype(int)
        df['timestep'] = df['timestep'].astype(int)
        df['time'] = df['timestep'] / num_inference_steps_classification
        df['std'] = df.groupby(['image_id', 'timestep'])['loss'].transform('std')

        return df

    # Create the DataFrame
    df = create_loss_dataframe(all_loss_saver_array, all_t, all_labels)

    median_std = df.groupby('time')['std'].median().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df, x='time', y='std', ax=ax1)
    ax1.set_title('Distribution of std across time (Box Plot)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Standard Deviation')

    # median plot with polynomial fits
    ax2.scatter(median_std['time'], median_std['std'], color='blue', label='median std')

    # Try polynomial fits of different degrees
    degrees = [4]
    colors = ['red', 'green', 'purple']

    for degree, color in zip(degrees, colors):
        # Fit polynomial
        coeffs = np.polyfit(median_std['time'], median_std['std'], degree)
        poly = np.poly1d(coeffs)

        # Plot fitted polynomial
        x_fit = np.linspace(median_std['time'].min(), median_std['time'].max(), 100)
        y_fit = poly(x_fit)
        ax2.plot(x_fit, y_fit, color=color, label=f'Degree {degree} fit')

        # Print polynomial coefficients in the required format
        print(f"\nDegree {degree} polynomial coefficients:")
        print(f"[a0, a1, ..., a{degree}] = {list(coeffs[::-1])}")
        logs[f"poly coeffs deg {degree}"] = list(coeffs[::-1])

    ax2.set_title('median std across time with Polynomial Fits')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('median Standard Deviation')
    ax2.legend()

    # Adjust the layout and display the plot
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = np.array(Image.open(buf))
    logs["poly fit to std"] = wandb.Image(image)

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            if args.validation_prompts:
                logs["images"] = [wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}") for i, image in enumerate(images_val_prompts)]

            # Randomly shuffle indices for unpredicted and predicted correctly images
            wrong_indices = [i for i, correct in enumerate(correct_tracker) if not correct]
            correct_indices = [i for i, correct in enumerate(correct_tracker) if correct]
            random.shuffle(wrong_indices)
            random.shuffle(correct_indices)

            # Select up to 15 wrong predictions and 10 correct predictions
            selected_wrong_indices = wrong_indices[:15]
            selected_correct_indices = correct_indices[:10]

            # Create logs for wrong and correct classification images
            logs["wrong_classification_images"] = [
                wandb.Image(all_classification_images[i], caption=f"{i}") for i in selected_wrong_indices
            ]
            logs["correct_classification_images"] = [
                wandb.Image(all_classification_images[i], caption=f"{i}") for i in selected_correct_indices
            ]
            logs["heatmap_images"] = [wandb.Image(image, caption=f"{i}: {unique_prompts_short[(i % n_classes)]}") for i, image in enumerate(all_heatmaps_images)]

            tracker.log(logs)

    torch.cuda.empty_cache()

    return total_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
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
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--n_image_rounds",
        type=int,
        default=30,
        help=("number of images for each class to use for validation"),
    )
    parser.add_argument(
        "--n_noise_lvls",
        type=int,
        default=30,
        help=("Maximum number forward passes (trials) to use for each image."),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str_to_float_list,
        default=None,
        nargs="+",
        help=("A set of float number arrays evaluated every `--validation_epochs` and logged to `--report_to`. Each array should be enclosed in quotes."),
    )
    parser.add_argument(
        "--unique_prompts_to_add",
        type=str_to_float_list,
        default=None,
        nargs="+",
        help=("A set of float number arrays that should be added to the unique prompts."),
    )
    parser.add_argument(
        "--poly_coeff",
        type=str_to_float_list,
        default=None,
        nargs="+",
        help=("the coefficients of the polynomial to use for validation weighting"),
    )
    parser.add_argument(
        "--class_prompts",
        type=str_to_float_list,
        default=None,
        nargs="+",
        help=("The class prompts to validate with if we are using unlabelled data."),
    )
    parser.add_argument(
        "--classes_to_skip_for_validation",
        type=str,
        default=None,
        nargs="+",
        help=("The classes to not validate"),
    )
    parser.add_argument(
        "--validation_img_dir",
        type=str,
        default=None,
        help=(
            "A folder containing validation images."
            " img2img is carried out on these images to asses if they're preserved. MSE in latent space logged to wandb."
        ),
    )

    parser.add_argument(
        "--w_weight_json_path",
        type=str,
        default=None,
        help=(
            "A folder containing validation json for the w_weight to calculate logistic regression."
        ),
    )
    parser.add_argument(
        "--validation_prompt_dir",
        type=str,
        default=None,
        help=(
            "A folder containing validation prompts in json files, needs the same name as the image file."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--min_step_to_validate",
        type=int,
        default=0,
        help=("won't validate until this step is reached"),
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=20,
        help=(
            "The radius for evaluation"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )

    parser.add_argument(
        "--random_rotation",
        action="store_true",
        help="whether to randomly rotate images by 0, 90, 180, 270 degrees, also flips diagonally with 50% probability",
    )

    parser.add_argument(
        "--random_full_rotation",
        action="store_true",
        help="whether to randomly rotate images by between 0 and 360 degrees",
    )

    parser.add_argument(
        "--color_jitter", action="store_true", help="whether to apply random brightness and contrast adjustments"
    )

    parser.add_argument(
        "--jitter_strength", type=float,  default=0.0, help="jitter strength (half for hue)"
    )

    parser.add_argument(
        "--random_gray_probability", type=float,  default=0.0, help="probability of turning the image to gray scale"
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--min_noise_trials",
        type=int,
        default=None,
        help="Use if you want to prune away unlikely classes, will start to prune after min_noise_trails.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--validate_at_the_end",
        default=False,
        help="Whether to validate the model on the validation set and the end of the training. Can be used with nr_steps = 0.",
    )

    parser.add_argument(
        "--save_at_end",
        default=True,
        help="Whether to save the model at the end of the training.",
    )
    parser.add_argument(
        "--random_augment",
        default=False,
        help="Whether to use random augmentations.",
    )
    parser.add_argument(
        "--random_augment_validation",
        default=False,
        help="Whether to use random augmentations.",
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.3,
        help="Mixup alpha to be used if mixup is enabled. "
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
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
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--hist_shift_prob", type=float, default=0.0, help="The probability of the histogram shift.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--n_heatmap_images",
        type=int,
        default=0,
        help="The number of heatmap images produced during validation.",
    )
    parser.add_argument(
        "--balance_data",
        type=bool,
        default=False,
        help=("Whether to balance the data by class."),
    )
    parser.add_argument(
        "--save_heatmaps_locally",
        type=bool,
        default=False,
        help=("Whether to save all the idividual heatmaps locally."),
    )
    parser.add_argument(
        "--use_exp_weighting",
        type=bool,
        default=False,
        help=("Whether to use exponential weighting during training."),
    )
    parser.add_argument(
        "--shuffle_images",
        type=bool,
        default=False,
        help=("Whether to shuffle the images during inference (Note, then the same image might be evaluated multiple times)."),
    )
    parser.add_argument(
        "--use_snr_weighting",
        type=bool,
        default=False,
        help=("to use snr weighting for the inference loss."),
    )
    parser.add_argument(
        "--save_all_wrong_images",
        type=bool,
        default=False,
        help=(""),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    vae.requires_grad_(False)
    unet.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if True:  # NOTE ALWAYS USING LOCAL DATA FOR NOW
        dataset = load_from_disk(args.dataset_name)
    else:
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                data_dir=args.train_data_dir,
            )
        else:
            data_files = {}
            if args.train_data_dir is not None:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    def convert_to_right_dim(examples):
        captions = []
        for caption in examples[caption_column]:
            captions.append(convert_dim(caption))

        # Convert the list of arrays into a torch tensor
        return torch.tensor(np.array(captions), dtype=weight_dtype)
        # return torch.tensor(captions, dtype=torch.float16)

    # Custom transform function for random rotation
    def random_rotation(image):
        angles = [0, 90, 180, 270]
        return TF.rotate(image, angle=random.choice(angles))

    def random_full_rotation(image):
        angles = np.linspace(0, 360, 360, endpoint=False)
        return TF.rotate(image, angle=random.choice(angles))

    def diagonal_mirror(image):
        if random.random() > 0.5:
            # Flip horizontally and then transpose
            return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
        return image


    jitter = args.jitter_strength
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            RandHistogramShift(args.hist_shift_prob),
            transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=(jitter/2)) if args.color_jitter else transforms.Lambda(lambda x: x),
            transforms.RandomGrayscale(p=args.random_gray_probability),
            transforms.Lambda(lambda x: RandAugment()(x)) if args.random_augment else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.Lambda(diagonal_mirror) if args.random_rotation else transforms.Lambda(lambda x: x),
            transforms.Lambda(random_rotation) if args.random_rotation else transforms.Lambda(lambda x: x),
            transforms.Lambda(random_full_rotation) if args.random_full_rotation else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = convert_to_right_dim(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # Move  vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def mixup_data(x, y, alpha=1.0):
        if alpha > 0.:
            lam = random.betavariate(alpha, alpha)
        else:
            lam = 1.

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y


    accuracy = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                mixed_x, mixed_y = mixup_data(batch["pixel_values"].to(weight_dtype), batch["input_ids"].to(weight_dtype), alpha=args.mixup_alpha)
                latents = vae.encode(mixed_x).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = mixed_y

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.use_exp_weighting:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape))))

                    def exponential(x):
                        a=16.30
                        b=-5.17
                        c=52
                        return a * torch.exp(-b * x) + c

                    balancing_factor = exponential((1000 - timesteps.float()) / 1000)
                    loss = loss / balancing_factor * 400  # 400 to get back to similar scale as before
                    loss = loss.mean()
                elif args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if args.min_step_to_validate <= global_step:
                        if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                            if args.use_ema:
                                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                                ema_unet.store(unet.parameters())
                                ema_unet.copy_to(unet.parameters())

                            accuracy = log_validation(
                                vae,
                                unet,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                                noise_scheduler,
                            )
                            if args.use_ema:
                                # Switch back to the original UNet parameters.
                                ema_unet.restore(unet.parameters())

                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: float(x.split("-")[2]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            str_accuracy = str(round(accuracy, 2))
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-{str_accuracy}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)


            if global_step >= args.max_train_steps:
                break

    if args.validate_at_the_end:
        if args.use_ema:
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())
        accuracy = log_validation(
            vae,
            unet,
            args,
            accelerator,
            weight_dtype,
            global_step,
            noise_scheduler,
        )
        if args.use_ema:
            # Switch back to the original UNet parameters.
            ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.save_at_end:
            unet = accelerator.unwrap_model(unet)
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())

            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
            ) # We don't use the text encoder in this code, but we want to save via the pipeline, which needs it as an argument.
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=args.revision,
                variant=args.variant,
                safety_checker=None,
            )
            pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
