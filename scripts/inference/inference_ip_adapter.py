"""
IP-Adapter inference script (experimental).

This script demonstrates an alternative approach using IP-Adapter
for image-guided editing. Note: This was an experimental approach
that was not used in the final implementation.
"""

import os
from datetime import datetime
from io import BytesIO
from typing import Optional

import PIL
import PIL.ImageOps
import requests
import torch
from PIL import Image

from diffusers import StableDiffusionInstructPix2PixPipeline


def download_image(url: str) -> Image.Image:
    """
    Download an image from a URL and convert it to RGB format.
    
    Args:
        url: URL of the image to download
        
    Returns:
        PIL Image in RGB format
    """
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def main():
    """Main inference function for IP-Adapter approach."""
    # Configuration
    model_id = "instruct-pix2pix-model"  # Update with your model path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "edited_images"
    ip_adapter_weights_path = "./ip-adapter_sd15_light.bin"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pipeline
    print(f"Loading model from {model_id}...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # Load IP-Adapter weights if available
    if pipe.image_encoder is not None and os.path.exists(ip_adapter_weights_path):
        print("Loading IP-Adapter weights...")
        pipe.image_encoder.load_state_dict(
            torch.load(ip_adapter_weights_path, map_location=device),
            strict=False
        )
        print("IP-Adapter weights loaded successfully.")
    else:
        print("Warning: IP-Adapter weights not found or image encoder not initialized.")
    
    # Example image URLs
    original_image_url = "https://pooperscoopers.motivatedbrands.ca/wp-content/uploads/sites/2/2024/08/A-dog-in-a-red-bucket-that-is-used-for-pet-waste-bucket-services-offered-by-Motivated-Pooper-Scoopers.png"
    object_image_url = "https://pbs.twimg.com/media/DlL5rKvW0AATU8n?format=jpg"
    
    # Download images
    print("Downloading images...")
    original_image = download_image(original_image_url)
    object_image = download_image(object_image_url)
    
    # Inference parameters
    num_inference_steps = 80
    image_guidance_scale = 1.9
    guidance_scale = 20.5
    
    # Run inference with IP-Adapter
    print("Generating edited image with IP-Adapter...")
    edited_image = pipe(
        prompt="add a zebra",  # Text prompt still used with IP-Adapter
        image=original_image,
        ip_adapter_image=object_image,  # IP-Adapter specific parameter
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
    ).images[0]
    
    # Save output
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = (
        f"edited_image_ip_adapter_{current_time}_"
        f"gs-{guidance_scale}_igs-{image_guidance_scale}_"
        f"nis-{num_inference_steps}.png"
    )
    file_path = os.path.join(output_dir, file_name)
    edited_image.save(file_path)
    
    print(f"Edited image saved at: {file_path}")


if __name__ == "__main__":
    main()
