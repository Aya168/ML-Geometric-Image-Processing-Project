"""
Inference script for image-guided object insertion.

This script demonstrates how to use the StableDiffusionInstructPix2PixImagePipeline
to edit images by inserting objects from reference images.
"""

import os
import sys
from datetime import datetime
from io import BytesIO
from typing import Optional

import requests
import torch
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.pipelines.pipeline_instruct_pix2pix_image import StableDiffusionInstructPix2PixImagePipeline


def download_image(url: str) -> Image.Image:
    """
    Download an image from a URL and convert it to RGB format.
    
    Args:
        url: URL of the image to download
        
    Returns:
        PIL Image in RGB format
    """
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def main():
    """Main inference function."""
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
    
    # Example image URLs - replace with your own
    original_image_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
    object_image_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"
    
    # Download images
    print("Downloading images...")
    original_image = download_image(original_image_url)
    object_image = download_image(object_image_url)
    
    # Inference parameters
    num_inference_steps = 100
    image_guidance_scale = 1.5  # Controls adherence to original image
    guidance_scale = 7.0  # Controls adherence to object image
    
    # Set scheduler timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
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
        f"nis-{num_inference_steps}.png"
    )
    file_path = os.path.join(output_dir, file_name)
    edited_image.save(file_path)
    
    print(f"Edited image saved at: {file_path}")


if __name__ == "__main__":
    main()
