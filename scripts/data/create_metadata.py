"""
Metadata creation script for organizing dataset into Hugging Face format.

This script organizes the prepared dataset into a structured format suitable
for uploading to Hugging Face Hub.
"""

import csv
import os
import shutil
from pathlib import Path
from typing import Optional

from tqdm import tqdm


# Configuration - update these paths as needed
MAIN_FOLDER = 'dataset_output'  # Path to folder containing image subfolders
OUTPUT_FOLDER = 'train'  # Output folder name
BASE_PATH_FOR_METADATA = "project_dataset_from_PIPE_train"


def organize_dataset(
    main_folder: str,
    output_folder: str,
    base_path_for_metadata: str
) -> None:
    """
    Organize dataset files and create metadata CSV.
    
    Args:
        main_folder: Path to folder containing image subfolders
        output_folder: Output folder name for organized dataset
        base_path_for_metadata: Base path prefix for metadata entries
    """
    # Create output directories
    object_image_folder = os.path.join(output_folder, 'object_image')
    target_image_folder = os.path.join(output_folder, 'target_image')
    original_image_folder = os.path.join(output_folder, 'original_image')

    os.makedirs(object_image_folder, exist_ok=True)
    os.makedirs(target_image_folder, exist_ok=True)
    os.makedirs(original_image_folder, exist_ok=True)

    # Initialize metadata list
    metadata = []

    # Get subfolders
    if not os.path.exists(main_folder):
        print(f"Error: Main folder not found at {main_folder}")
        print("Please update MAIN_FOLDER in the script.")
        return

    subfolders = [
        f for f in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, f))
    ]

    # Process subfolders with progress bar
    for subfolder_name in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_path = os.path.join(main_folder, subfolder_name)
        img_id = subfolder_name

        # Define source file paths
        object_image_src = os.path.join(subfolder_path, f'object_image_{img_id}.jpg')
        target_image_src = os.path.join(subfolder_path, f'target_image_{img_id}.jpg')
        original_image_src = os.path.join(subfolder_path, f'original_image_{img_id}.jpg')

        # Define destination paths
        object_image_dest = os.path.join(object_image_folder, f'{img_id}.jpg')
        target_image_dest = os.path.join(target_image_folder, f'{img_id}.jpg')
        original_image_dest = os.path.join(original_image_folder, f'{img_id}.jpg')

        # Log metadata
        metadata.append({
            "img_id": img_id,
            "original_image": (
                f"original_image/{img_id}.jpg" if os.path.exists(original_image_src)
                else "missing_file"
            ),
            "target_image": (
                f"target_image/{img_id}.jpg" if os.path.exists(target_image_src)
                else "missing_file"
            ),
            "object_image": (
                f"object_image/{img_id}.jpg" if os.path.exists(object_image_src)
                else "missing_file"
            )
        })

        # Copy files to their respective folders
        if os.path.exists(original_image_src):
            shutil.copy(original_image_src, original_image_dest)
        if os.path.exists(target_image_src):
            shutil.copy(target_image_src, target_image_dest)
        if os.path.exists(object_image_src):
            shutil.copy(object_image_src, object_image_dest)

    # Write metadata to CSV
    metadata_file = os.path.join(output_folder, 'metadata.csv')
    if metadata:
        with open(metadata_file, 'w', newline='') as csvfile:
            fieldnames = ["img_id", "original_image", "target_image", "object_image"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata)
        print(f"\nMetadata CSV successfully created at {metadata_file}.")
    else:
        print("No metadata to write.")


def main():
    """Main function."""
    organize_dataset(MAIN_FOLDER, OUTPUT_FOLDER, BASE_PATH_FOR_METADATA)


if __name__ == "__main__":
    main()
