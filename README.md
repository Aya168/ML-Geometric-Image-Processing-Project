# ML Geometric Image Processing Project

## Overview

This project implements an advanced image editing method based on diffusion models. Traditional InstructPix2Pix methods rely on textual instructions to edit images (e.g., "add a dog"). Our method extends this by allowing users to substitute textual descriptions with **reference images** of the desired objects. This enables more precise object insertion without requiring manual annotation or complex text prompt engineering.

The core innovation is an **Embedding Optimizer** that projects the CLIP embeddings of a reference object image into the textual embedding space of Stable Diffusion, allowing the model to "read" the object image as if it were a text instruction.

## Key Features

-   **Image-Based Instruction**: Edit images using a reference object image instead of text.
-   **No Manual Annotation**: The insertion process is handled by the model without needing user-provided masks or coordinates.
-   **Custom Embedding Optimization**: Features a trained MLP-based adapter that translates visual features into textual conditioning tokens.
-   **Fine-tuned InstructPix2Pix**: Based on the InstructPix2Pix architecture, adapted to accept multi-modal inputs.

## Project Structure

The project is organized as follows:

```
project_root/
├── src/                        # Source code for pipelines and models
│   ├── pipelines/              # Custom diffusion pipelines
│   │   └── pipeline_instruct_pix2pix_image.py  # Main pipeline for image-guided editing
│   └── models/                 # Model definitions
│       └── embedding_optimizer.py # Embedding Optimizer architecture
├── scripts/                    # Executable scripts
│   ├── training/               # Training scripts
│   │   └── train.py            # Main training script (InstructPix2Pix + EmbeddingOptimizer)
│   ├── inference/              # Inference scripts
│   │   ├── inference.py        # Main inference script
│   │   └── inference_local.py  # Local inference testing
│   ├── data/                   # Data preparation scripts
│   │   ├── prepare_dataset.py  # Dataset cropping/preparation
│   │   └── create_metadata.py  # Metadata generation
│   └── utils/                  # Utilities
│       └── upload_to_hub.py    # Hugging Face Hub uploader
├── docs/                       # Documentation
└── requirements.txt            # Dependencies
```

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Inference

To run inference, use `scripts/inference/inference.py`. You will need to specify the URL or path to your input image (`image`) and the reference object image (`ob_image`).

```python
# Example usage inside scripts/inference/inference.py
url = "path_to_original_image.jpg"
url2 = "path_to_object_image.jpg"

# ... pipeline setup ...

edited_image = pipe(
    prompt="",  # Prompt is replaced by object image embeddings
    image=image,
    ob_image=image2,
    num_inference_steps=100,
    image_guidance_scale=1.5,
    guidance_scale=7.0
).images[0]
```

Run the script:
```bash
python scripts/inference/inference.py
```

### 2. Training

To train the model on your own dataset:

```bash
python scripts/training/train.py \
    --pretrained_model_name_or_path="paint-by-inpaint/add-base" \
    --dataset_name="ADT1999/project_from_PIPE_extended" \
    --output_dir="instruct-pix2pix-model-final" \
    --resolution=224 \
    --train_batch_size=16 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --val_image_url="<URL_TO_VAL_IMAGE>" \
    --val_image_url2="<URL_TO_VAL_OBJECT>"
```

### 3. Data Preparation

If you want to recreate the dataset from the PIPE dataset:

1.  Run `scripts/data/prepare_dataset.py` to download and crop images.
2.  Run `scripts/data/create_metadata.py` to structure the data and generate `metadata.csv`.

## Models & Resources

-   **Fine-tuned Model**: [ADT1999/instruct-pix2pix-model-final](https://huggingface.co/ADT1999/instruct-pix2pix-model-final)
-   **Dataset**: [ADT1999/project_from_PIPE_extended](https://huggingface.co/ADT1999/project_from_PIPE_extended)
