"""
Utility script to check cosine similarity between CLIP image and text embeddings.

This script is useful for validating the alignment between image and text
embedding spaces, which is important for the Embedding Optimizer.
"""

import torch
from io import BytesIO
from typing import List

import requests
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)


def download_image_from_url(url: str) -> Image.Image:
    """
    Download an image from a URL.
    
    Args:
        url: URL of the image
        
    Returns:
        PIL Image object
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def compute_cosine_similarity(
    image_url: str,
    text_inputs: List[str],
    model_name: str = "openai/clip-vit-large-patch14",
    similarity_threshold: float = 0.8
) -> None:
    """
    Compute cosine similarity between image and text embeddings.
    
    Args:
        image_url: URL of the image to analyze
        text_inputs: List of text descriptions to compare
        model_name: CLIP model name
        similarity_threshold: Threshold for determining similarity
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    print(f"Loading CLIP models ({model_name})...")
    vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
    text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # Load and preprocess image
    print(f"Loading image from {image_url}...")
    image = download_image_from_url(image_url)
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Tokenize text inputs
    tokenized_text = tokenizer(text_inputs, padding=True, return_tensors="pt").to(device)

    # Get embeddings
    print("Computing embeddings...")
    with torch.no_grad():
        # Generate image embeddings
        image_embeddings = vision_model(**inputs).image_embeds  # Shape: [1, 1024]
        
        # Generate text embeddings
        text_outputs = text_model(**tokenized_text)
        text_embeddings = text_outputs.text_embeds  # Shape: [num_texts, 1024]

    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")

    # Compute cosine similarity
    image_embeddings = image_embeddings.squeeze(0)  # Shape: [1024]
    cosine_similarities = torch.nn.functional.cosine_similarity(
        image_embeddings.unsqueeze(0), text_embeddings, dim=1
    )

    # Print results
    print("\nCosine Similarity Results:")
    print("-" * 50)
    for text, similarity in zip(text_inputs, cosine_similarities):
        is_similar = similarity.item() > similarity_threshold
        status = "✓ Similar" if is_similar else "✗ Not similar"
        print(f"{text:30s}: {similarity.item():.4f} [{status}]")
    
    print(f"\nThreshold: {similarity_threshold}")


def main():
    """Main function with example usage."""
    # Example configuration
    image_url = "https://www.mansfieldtexas.gov/ImageRepository/Document?documentId=6501"
    text_inputs = ["dog in a park", "cat", "dog", "cat", "ant"]
    
    compute_cosine_similarity(
        image_url=image_url,
        text_inputs=text_inputs,
        similarity_threshold=0.8
    )


if __name__ == "__main__":
    main()
