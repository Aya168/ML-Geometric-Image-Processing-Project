"""
Dataset preparation script for creating object insertion dataset from PIPE.

This script processes the PIPE and PIPE_Masks datasets to create triplets of
(original_image, object_image, target_image) by cropping objects based on masks.
"""

import io
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image


# Configuration - update these paths as needed
OUTPUT_DIR = 'dataset_output'
SAMPLE_SIZE = 120000
BATCH_SIZE = 1000
METADATA_CSV_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_pil_image(data) -> Image.Image:
    """
    Ensure the input is a PIL Image, converting if necessary.
    
    Args:
        data: Image data (PIL Image, bytes, or bytearray)
        
    Returns:
        PIL Image object
        
    Raises:
        TypeError: If data type is not supported
    """
    if isinstance(data, Image.Image):
        return data
    elif isinstance(data, (bytes, bytearray)):
        return Image.open(io.BytesIO(data))
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a binary mask to a bounding box.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Tuple of (x_min, y_min, width, height) or None if mask is empty
    """
    mask_tensor = torch.tensor(mask, device=DEVICE)
    rows = torch.any(mask_tensor, dim=1)
    cols = torch.any(mask_tensor, dim=0)

    if not torch.any(rows) or not torch.any(cols):
        return None

    y_min, y_max = torch.where(rows)[0][[0, -1]].cpu().numpy()
    x_min, x_max = torch.where(cols)[0][[0, -1]].cpu().numpy()

    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))


def normalize_bbox(mask_shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Normalize bounding box coordinates to 512x512 resolution.
    
    Args:
        mask_shape: Original mask shape (height, width)
        bbox: Bounding box (x_min, y_min, width, height)
        
    Returns:
        Normalized bounding box
    """
    x_min, y_min, width, height = bbox
    mask_height, mask_width = mask_shape

    x_scale = 512 / mask_width
    y_scale = 512 / mask_height

    return (
        int(x_min * x_scale),
        int(y_min * y_scale),
        int(width * x_scale),
        int(height * y_scale)
    )


def crop_image_by_bbox(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop an image using a bounding box.
    
    Args:
        image: PIL Image to crop
        bbox: Bounding box (x_min, y_min, width, height)
        
    Returns:
        Cropped PIL Image
    """
    x_min, y_min, width, height = bbox
    return image.crop((x_min, y_min, x_min + width, y_min + height))


def save_cropped_image(
    image_id: str,
    bbox: Tuple[int, int, int, int],
    target_image_data,
    original_image_data,
    metadata: list,
    output_dir: str
) -> None:
    """
    Save cropped images and update metadata.
    
    Args:
        image_id: Unique identifier for the image
        bbox: Bounding box for cropping
        target_image_data: Target image data
        original_image_data: Original image data
        metadata: Metadata list to append to
        output_dir: Output directory path
    """
    try:
        # Ensure images are PIL Images
        target_image = ensure_pil_image(target_image_data)
        original_image = ensure_pil_image(original_image_data)

        # Crop target image using the bounding box to get object image
        object_image = crop_image_by_bbox(target_image, bbox)

        # Create image directory
        image_dir = os.path.join(output_dir, str(image_id))
        os.makedirs(image_dir, exist_ok=True)

        # Define file paths
        object_image_path = os.path.join(image_dir, f"object_image_{image_id}.jpg")
        target_image_path = os.path.join(image_dir, f"target_image_{image_id}.jpg")
        original_image_path = os.path.join(image_dir, f"original_image_{image_id}.jpg")

        # Save images
        object_image.save(object_image_path)
        target_image.save(target_image_path)
        original_image.save(original_image_path)

        # Append metadata
        metadata.append({
            'id': str(image_id),
            'object_image': object_image_path,
            'target_image': target_image_path,
            'original_image': original_image_path
        })

        print(f"Cropped images saved for {image_id}.")
    except Exception as e:
        print(f"Error processing {image_id}: {e}")


def main():
    """Main processing function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    masks_dataset = load_dataset('paint-by-inpaint/PIPE_Masks', split='train')
    images_dataset = load_dataset('paint-by-inpaint/PIPE', split='train')

    selected_columns = ['img_id', 'target_img', 'source_img']
    metadata = []

    # Process in batches
    for i in range(0, SAMPLE_SIZE, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, SAMPLE_SIZE)
        print(f"Processing batch {i} to {batch_end}...")

        # Load batch
        batch = images_dataset.select(range(i, batch_end))
        batch = batch.remove_columns([col for col in batch.column_names if col not in selected_columns])
        batch_df = pd.DataFrame(batch)

        # Process each mask and associated image
        for idx, mask_data in enumerate(masks_dataset.select(range(i, batch_end))):
            try:
                image_id = mask_data['img_id']
                mask = ensure_pil_image(mask_data['mask']).convert('L')
                mask_np = np.array(mask)

                # Get bounding box
                bbox = mask_to_bbox(mask_np)
                if bbox:
                    bbox = normalize_bbox(mask_np.shape, bbox)

                    # Fetch image data
                    row = batch_df.loc[batch_df['img_id'] == image_id]
                    if row.empty:
                        print(f"No matching image found for {image_id}.")
                        continue

                    target_image_data = row.iloc[0]['target_img']
                    original_image_data = row.iloc[0]['source_img']

                    # Save cropped images
                    save_cropped_image(
                        image_id, bbox, target_image_data, original_image_data,
                        metadata, OUTPUT_DIR
                    )
            except Exception as e:
                print(f"Skipping mask {mask_data.get('img_id', 'unknown')} due to error: {e}")

        # Write metadata incrementally
        if metadata:
            pd.DataFrame(metadata).to_csv(
                METADATA_CSV_PATH,
                mode='a',
                header=not os.path.exists(METADATA_CSV_PATH),
                index=False
            )
            metadata.clear()

    print("Processing complete.")


if __name__ == "__main__":
    main()
