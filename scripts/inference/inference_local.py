"""
Local inference script for testing with local image files.

This script loads images from local file paths instead of URLs,
useful for testing with your own dataset.
"""

import os
import sys
from datetime import datetime
from typing import Optional

import torch
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.pipelines.pipeline_instruct_pix2pix_image import StableDiffusionInstructPix2PixImagePipeline


def load_image_from_path(image_path: str) -> Image.Image:
    """
    Load an image from a local file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image in RGB format
    """
    image = Image.open(image_path).convert("RGB")
    return image


def main():
    """Main inference function for local images."""
    # Configuration
    model_id = "ADT1999/instruct-pix2pix-model-final"  # Update with your model path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "edited_images"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pipeline
    print(f"Loading model from {model_id}...")
    pipe = StableDiffusionInstructPix2PixImagePipeline.from_pretrained(
        model_id,
        device_type=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # Example: Update these paths to point to your local images
    # Format: original_image_path should point to the base image
    #         object_image_path should point to the object to insert
    image_id = "example_id"  # Replace with your image ID
    original_image_path = f"path/to/original_image_{image_id}.jpg"
    object_image_path = f"path/to/object_image_{image_id}.jpg"
    
    # Load images
    print("Loading images from local paths...")
    if not os.path.exists(original_image_path):
        print(f"Error: Original image not found at {original_image_path}")
        print("Please update original_image_path in the script.")
        return
    
    if not os.path.exists(object_image_path):
        print(f"Error: Object image not found at {object_image_path}")
        print("Please update object_image_path in the script.")
        return
    
    original_image = load_image_from_path(original_image_path)
    object_image = load_image_from_path(object_image_path)
    
    # Inference parameters
    num_inference_steps = 100
    image_guidance_scale = 1.5
    guidance_scale = 7.0
    
    # Run inference
    print("Generating edited image...")
    edited_image = pipe(
        prompt="",  # Empty prompt - object image replaces text conditioning
        image=original_image,
        object_image=object_image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
    ).images[0]
    
    # Save output
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = (
        f"edited_image_{current_time}_"
        f"gs-{guidance_scale}_igs-{image_guidance_scale}_"
        f"nis-{num_inference_steps}_id-{image_id}.png"
    )
    file_path = os.path.join(output_dir, file_name)
    edited_image.save(file_path)
    
    print(f"Edited image saved at: {file_path}")


if __name__ == "__main__":
    main()
