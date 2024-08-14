import os
import shutil
import numpy as np
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from PIL import Image
import json
import gc
import time
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms

IMAGE_RESIZE_SIZE = 360  # Variable for image resizing

def split_into_chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def check_and_rename_folder(source_path):
    # NOTE: we use 10 classes, and merge some classes according to the following mapping. Feel free to modify this.
    n_classes = 10
    name_to_number = {
        "basophil": 0, "blast": 1, "lymphoblast": 1, "myeloblast": 1, "lymphocyte": 2, "monocyte": 3,
        "neutrophil band": 4, "neutrophil segment": 4, "neutrophil": 4,
        "normoblast": 5, "erythroblast": 5, "eosinophil": 6,
        "immature granulocyte": 7, "ig": 7, "platelet": 8, "artefact": 9,
        "BAS": 0, "EBO": 5, "EOS": 6, "KSC": 9, "LYA": 2, "LYT": 2, "MMZ": 7,
        "MOB": 1, "MON": 3, "MYB": 7, "MYO": 1, "NGB": 4, "NGS": 4, "PMB": 7, "PMO": 7,
    }

    folder_to_number = {str(i): i for i in range(n_classes)}
    dimensions = len(folder_to_number)

    source_path = Path(source_path)
    if not source_path.is_dir():
        print(f"Error: The provided path '{source_path}' is not a directory.")
        return None, None

    print(f"Checking folder structure of '{source_path}'...")

    subfolders = [f.name.lower() for f in source_path.iterdir() if f.is_dir()]
    invalid_subfolders = [f for f in subfolders if f not in name_to_number]

    if invalid_subfolders:
        print(f"Error: Found invalid subfolders: {', '.join(invalid_subfolders)}")
        print(f"Must be one of: {', '.join(name_to_number.keys())}")
        print("Please update the name_to_number dictionary or the name of your folders")
        print("Renaming process aborted.")
        return None, None

    print("All subfolders are valid. Proceeding with renaming...")

    target_path = source_path.with_name(f"{source_path.name}_train_val_test")
    target_path.mkdir(exist_ok=True)
    print(f"Created target directory for renamed images: '{target_path}'")

    json_path = target_path / "_json"
    json_path.mkdir(exist_ok=True)
    print(f"Created target directory for JSON files: '{json_path}'")

    def one_hot_encode(value):
        vector = [0] * dimensions
        assert value < dimensions
        vector[value] = 1
        return vector

    def save_json_file(path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    counter = 0
    for subfolder in source_path.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name.lower()
            new_name = str(name_to_number[subfolder_name])
            new_subfolder = target_path / new_name
            new_subfolder.mkdir(exist_ok=True)
            print(f"Created subfolder: '{new_subfolder}'")

            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', ".tiff", ".tif")
            images = [f for f in subfolder.iterdir() if f.suffix.lower() in image_extensions]

            print(f"Found {len(images)} images in {subfolder_name}")

            for image in tqdm(images, desc=f"Processing {subfolder_name}"):
                # Create a unique filename by prepending the original subfolder name
                unique_filename = f"{subfolder_name}_{image.name}"
                new_image_path = new_subfolder / unique_filename
                shutil.copy2(image, new_image_path)
                counter += 1

                category_number = folder_to_number[new_name]
                encoded_vector = one_hot_encode(category_number)
                json_file_path = json_path / new_name / f"{unique_filename.split('.')[0]}.json"
                save_json_file(json_file_path, encoded_vector)

    if counter == 0:
        print("ERROR: No images found in the subfolders of the source path.")
        return None, None
    else:
        print(f"Total images processed: {counter}")

    print("Folder structure checking, renaming, and JSON file creation completed successfully.")
    return target_path, json_path

def get_class_sizes(source_dir):
    class_sizes = {}
    for cls in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, cls)
        if os.path.isdir(class_dir):
            files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            class_sizes[cls] = len(files)
    return class_sizes

def create_splits_percentage(source_dir, train_percent, val_percent):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    class_sizes = get_class_sizes(source_dir)
    print(f"Class sizes: {class_sizes}")

    test_percent = 1 - train_percent - val_percent
    print(f"Test split will be: {test_percent*100:.2f}%")

    for cls in classes:
        if class_sizes[cls] == 0:
            print(f"Skipping class {cls} as it contains no images.")
            continue

        print(f"Processing class {cls}")
        class_dir = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        np.random.shuffle(files)

        num_train = int(len(files) * train_percent)
        num_val = int(len(files) * val_percent)
        num_test = len(files) - num_train - num_val

        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]

        print(f"Num train files: {len(train_files)}, Num val files: {len(val_files)}, Num test files: {len(test_files)}")

        move_files(source_dir, cls, train_files, val_files, test_files)

def create_splits_fixed(source_dir, train_count, val_count, test_count):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    class_sizes = get_class_sizes(source_dir)
    print(f"Class sizes: {class_sizes}")

    for cls in classes:
        if class_sizes[cls] == 0:
            print(f"Skipping class {cls} as it contains no images.")
            continue

        print(f"Processing class {cls}")
        class_dir = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        np.random.shuffle(files)

        total_required = train_count + val_count + test_count
        if len(files) < total_required:
            print(f"Warning: Not enough images in class {cls}. Using all available images.")
            train_files = files[:min(train_count, len(files))]
            val_files = files[len(train_files):min(len(train_files) + val_count, len(files))]
            test_files = files[len(train_files) + len(val_files):]
        else:
            train_files = files[:train_count]
            val_files = files[train_count:train_count + val_count]
            test_files = files[train_count + val_count:train_count + val_count + test_count]

        print(f"Num train files: {len(train_files)}, Num val files: {len(val_files)}, Num test files: {len(test_files)}")

        move_files(source_dir, cls, train_files, val_files, test_files)


def move_files(source_dir, cls, train_files, val_files, test_files):
    def move_split(files, split_type):
        split_dir = os.path.join(source_dir, split_type, cls)
        os.makedirs(split_dir, exist_ok=True)
        class_dir = os.path.join(source_dir, cls)
        for file in tqdm(files, desc=f"Moving {split_type} files for {cls}"):
            shutil.move(os.path.join(class_dir, file), os.path.join(split_dir, file))

    move_split(train_files, 'train')
    move_split(val_files, 'val')
    move_split(test_files, 'test')

    original_class_dir = os.path.join(source_dir, cls)
    if not os.listdir(original_class_dir):
        os.rmdir(original_class_dir)

def get_image_files(root, numeric_folder_path):
    for path, subdirs, files in os.walk(root):
        if "_json" in path:
            print(f"Skipping {path}")
            continue

        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")):
                relative_path = os.path.relpath(path, root)
                json_file_path = os.path.join(numeric_folder_path, relative_path, name.split(".")[0] + ".json")

                if os.path.exists(json_file_path):
                    yield os.path.join(path, name), json_file_path
                else:
                    print(f"NOT Found {json_file_path}")

def load_image(file_path, json_path, image_h_and_w):
    img = Image.open(file_path)
    width, height = img.size

    if width > 2000 or height > 2000:
        print(f"Skipping {file_path} due to large size (dimensions: {width}x{height})")
        return None, None, None

    img = img.convert("RGB").resize((image_h_and_w, image_h_and_w))

    with open(json_path, 'r') as f:
        numeric = json.load(f)
        numeric = np.array(numeric, dtype=np.float32)

    return img, numeric

def process_images(image_files, image_h_and_w):
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(load_image, file_path, json_path, image_h_and_w): (file_path, json_path)
            for file_path, json_path, in image_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
            result = future.result()
            if result[0] is not None:
                results.append(result)
    return results

def create_arrow_data(train_dir, numeric_folder_path, image_h_and_w=IMAGE_RESIZE_SIZE, chunk_size=30000):
    initial_time = time.time()

    # Create the Arrow data folder in the same directory as the train_dir
    base_path = os.path.join(os.path.dirname(train_dir), f"data_arrow")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    image_files = list(get_image_files(train_dir, numeric_folder_path))
    random.shuffle(image_files)
    print(f"Total images found: {len(image_files)}")

    chunks = list(split_into_chunks(image_files, chunk_size))

    for index, chunk in enumerate(chunks):
        dataset_path = os.path.join(base_path, f"dataset_chunk_{index + 1}")

        if os.path.exists(dataset_path):
            print(f"Skipping already processed chunk {index + 1}")
            continue

        processed_images_texts = process_images(chunk, image_h_and_w)
        approximated_time_it_will_take = len(chunk) * 1 / 60000
        print(f"Creating .arrow file {index+1}/{len(chunks)}. This will take a while (~ {approximated_time_it_will_take:.2f} hours.)")

        data = {
            "image": [img for img, text in processed_images_texts],
            "text": [text for img, text in processed_images_texts],
        }

        dataset = Dataset.from_dict(data)
        dataset_dict = DatasetDict({"train": dataset})
        dataset_dict.save_to_disk(dataset_path)

        print(f"Time taken: {time.time() - initial_time}")
        print(f"Number of images processed: {(index + 1) * chunk_size}")

        del processed_images_texts, data, dataset, dataset_dict
        gc.collect()

    n_chunks = len(chunks)

    all_datasets = []
    for i in range(1, n_chunks + 1):
        dataset = load_from_disk(os.path.join(base_path, f"dataset_chunk_{i}"))["train"]
        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        merged_train_dataset = all_datasets[0]
    else:
        merged_train_dataset = concatenate_datasets(all_datasets)

    merged_dataset_dict = DatasetDict({"train": merged_train_dataset})
    merged_dataset_dict.save_to_disk(os.path.join(base_path, "dataset_with_text_MERGED"))

    print(f"Arrow data created and saved in: {base_path}")

# --- Main Execution ---
if __name__ == "__main__":
    source_path = input("Enter the path to the folder containing subfolders with images: ")
    source_path = Path(source_path)
    renamed_path, json_path = check_and_rename_folder(source_path)

    if renamed_path and json_path:
        class_sizes = get_class_sizes(renamed_path)

        split_method = input("Choose splitting method (1 for percentage-based, 2 for fixed number per class): ")

        if split_method == "1":
            train_percent = float(input("Enter the percentage for train split (e.g. 70): ")) / 100
            val_percent = float(input("Enter the percentage for validation split (the rest will be used in the test set) (e.g. 10): ")) / 100
            create_splits_percentage(renamed_path, train_percent, val_percent)
        elif split_method == "2":
            train_count = int(input("Enter the number of images per class for train split (e.g. 50): "))
            val_count = int(input("Enter the number of images per class for validation split (e.g. 150): "))
            test_count = int(input("Enter the number of images per class for test split (e.g. 150): "))
            create_splits_fixed(renamed_path, train_count, val_count, test_count)
        else:
            print("Invalid choice. Exiting.")
            exit()

        print("Dataset splitting completed successfully.")

        train_dir = os.path.join(renamed_path, "train")
        create_arrow_data(train_dir, json_path)
        print("Arrow data creation completed successfully.")
    else:
        print("Process aborted due to errors in folder structure or renaming.")
