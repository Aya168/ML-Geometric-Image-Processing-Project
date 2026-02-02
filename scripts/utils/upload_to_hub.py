"""
Script to upload dataset to Hugging Face Hub.

This script prepares and uploads the dataset in the format required
by Hugging Face Hub for use in training.
"""

import os
from typing import Optional

import pandas as pd
from datasets import Dataset, Features, Image, Value


# Configuration - update these paths as needed
BASE_DIR = "./train"  # Base directory containing the dataset
CSV_FILE = "metadata.csv"  # Metadata CSV file name
REPO_NAME = "your-username/your-dataset-name"  # Hugging Face Hub repository name


def upload_dataset_to_hub(
    base_dir: str,
    csv_file: str,
    repo_name: str
) -> None:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        base_dir: Base directory where dataset is located
        csv_file: Name of the metadata CSV file
        repo_name: Hugging Face Hub repository name (format: username/dataset-name)
    """
    csv_path = os.path.join(base_dir, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please update BASE_DIR and CSV_FILE in the script.")
        return
    
    # Load CSV file
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Update paths in DataFrame to include full path
    df['original_image'] = df['original_image'].apply(
        lambda x: os.path.join(base_dir, x) if not os.path.isabs(x) else x
    )
    df['target_image'] = df['target_image'].apply(
        lambda x: os.path.join(base_dir, x) if not os.path.isabs(x) else x
    )
    df['object_image'] = df['object_image'].apply(
        lambda x: os.path.join(base_dir, x) if not os.path.isabs(x) else x
    )
    
    # Define dataset features
    features = Features({
        'img_id': Value('string'),
        'original_image': Image(),
        'target_image': Image(),
        'object_image': Image()
    })
    
    # Create dataset from DataFrame
    print("Creating dataset from DataFrame...")
    dataset = Dataset.from_pandas(df, features=features)
    
    # Remove any unnecessary columns
    if "_index_level_0_" in dataset.column_names:
        dataset = dataset.remove_columns("_index_level_0_")
    
    # Push to Hugging Face Hub
    print(f"Uploading dataset to {repo_name}...")
    dataset.push_to_hub(repo_name)
    
    print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_name}")


def main():
    """Main function."""
    upload_dataset_to_hub(BASE_DIR, CSV_FILE, REPO_NAME)


if __name__ == "__main__":
    main()
