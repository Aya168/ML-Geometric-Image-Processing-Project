# Visually Guided Object Insertion Into Image

**Authors:** Adi Tsach & Aya Spira  
**Supervised by:** Noam Rotstein & Roy Gantz

## Overview

This project implements an advanced image editing method based on diffusion models. Traditional InstructPix2Pix methods rely on textual instructions to edit images (e.g., "add a dog"). Our method extends this by allowing users to substitute textual descriptions with **reference images** of the desired objects. This enables more precise object insertion without requiring manual annotation or complex text prompt engineering.

Unlike existing image-conditioned methods (like Paint by Example) that require user-provided masks to designate insertion points, our approach eliminates the need for manual marking by replacing the textual input with an object image as the conditioning mechanism. This retains the model's generative strengths and realistic outputs while broadening its applicability to scenarios where a visual example is more intuitive or descriptive than text.

## Key Features

-   **Image-Based Instruction**: Edit images using a reference object image instead of text prompts.
-   **No Manual Annotation**: The insertion process is handled by the model without needing user-provided masks or coordinates.
-   **Custom Embedding Optimization**: Features a trained MLP-based adapter (Embedding Optimizer) that translates visual features from CLIP image embeddings into textual conditioning tokens compatible with Stable Diffusion.
-   **Fine-tuned InstructPix2Pix**: Based on the InstructPix2Pix architecture, adapted to accept multi-modal inputs through cross-attention layer training.

## Architecture

### Embedding Optimizer

The core innovation is an **Embedding Optimizer** that projects CLIP embeddings of a reference object image into the textual embedding space of Stable Diffusion. The architecture consists of:

1. **Input**: CLIP image embeddings of shape `[batch_size, 257, 1024]`
2. **Projection Layer**: Reduces dimensions from 1024 → 768
3. **Attention Pooling**: Reduces sequence length from 257 → 77 (matching Stable Diffusion's context length)
4. **MLP Refinement Block**: Four fully connected layers with ReLU activations (FC1, FC2, FC3, FC4)
5. **Output**: Embeddings of shape `[batch_size, 77, 768]` compatible with UNet cross-attention layers

This allows the model to "read" the object image as if it were a text instruction, enabling seamless integration into the existing diffusion pipeline.

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
│   │   ├── train.py            # Main training script (InstructPix2Pix + EmbeddingOptimizer)
│   │   └── train_ip_adapter.py # IP-Adapter variant training
│   ├── inference/              # Inference scripts
│   │   ├── inference.py        # Main inference script
│   │   ├── inference_local.py  # Local inference testing
│   │   └── inference_ip_adapter.py # IP-Adapter inference
│   ├── data/                   # Data preparation scripts
│   │   ├── prepare_dataset.py  # Dataset cropping/preparation from PIPE
│   │   └── create_metadata.py  # Metadata generation
│   └── utils/                  # Utilities
│       ├── check_cosine_similarity.py # CLIP embedding similarity validation
│       └── upload_to_hub.py    # Hugging Face Hub uploader
├── docs/                       # Documentation
│   └── Project Presentation.pdf # Project presentation slides
└── requirements.txt            # Dependencies
```

## Installation

1.  Clone the repository:
```bash
git clone https://github.com/Aya168/ML-Geometric-Image-Processing-Project.git
cd ML-Geometric-Image-Processing-Project
```

2.  Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Inference

To run inference, use `scripts/inference/inference.py`. You will need to specify the URL or path to your input image (`image`) and the reference object image (`object_image`).

```python
# Example usage inside scripts/inference/inference.py
url = "path_to_original_image.jpg"
url2 = "path_to_object_image.jpg"

# ... pipeline setup ...

edited_image = pipe(
    prompt="",  # Prompt is replaced by object image embeddings
    image=image,
    object_image=image2,
    num_inference_steps=100,
    image_guidance_scale=1.5,  # Controls adherence to original image
    guidance_scale=7.0          # Controls adherence to object image
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
    --validation_original_image_url="<URL_TO_ORIGINAL_IMAGE>" \
    --validation_object_image_url="<URL_TO_OBJECT_IMAGE>"
```

**Key Training Components:**
- **Embedding Optimizer**: Trained with full learning rate
- **UNet Cross-Attention Layers**: Trained with 0.1x learning rate (only specific blocks: down_blocks.4, mid_block, up_blocks.0)
- **VAE & Image Encoder**: Frozen during training

### 3. Data Preparation

If you want to recreate the dataset from the PIPE dataset:

1.  Run `scripts/data/prepare_dataset.py` to download and crop images from PIPE and PIPE_Masks datasets.
2.  Run `scripts/data/create_metadata.py` to structure the data and generate `metadata.csv`.

The dataset consists of triplets: `(original_image, object_image, target_image)` where:
- `original_image`: The base image to be edited
- `object_image`: The cropped object to be inserted (extracted using masks)
- `target_image`: The desired result with the object inserted

## Development Process

This project evolved through several iterations:

1. **Step 1 - CLIP**: Initial attempt using CLIPVisionModel directly (failed - insufficient semantic alignment)
2. **Step 2 - Paint by Example**: Attempted to use Paint by Example's image encoder (failed - architecture mismatch)
3. **Step 3 - IP Adapter**: Tried IP-Adapter module (failed - limited semantic capture)
4. **Step 4 - Dataset Creation**: Created custom dataset from PIPE (42,000 → 108,093 triplets)
5. **Step 5 - Training & Adaptation**: Developed Embedding Optimizer and fine-tuned cross-attention layers (successful)

## Models & Resources

-   **Fine-tuned Model**: [ADT1999/instruct-pix2pix-model-final](https://huggingface.co/ADT1999/instruct-pix2pix-model-final)
-   **Extended Dataset**: [ADT1999/project_from_PIPE_extended](https://huggingface.co/ADT1999/project_from_PIPE_extended) (108,093 triplets)
-   **Initial Dataset**: [Aya168/project_from_PIPE](https://huggingface.co/datasets/Aya168/project_from_PIPE) (42,000 triplets)

## Limitations & Future Work

### Current Limitations

- **Semantic Bridging**: The gap between text and image embeddings still presents challenges in some scenarios
- **Classifier-Free Guidance**: Optimal settings vary per case and require manual tuning
- **Generalization**: Further expansion of training resources and dataset diversity needed
- **Mask Generation**: Potential alternative approach using segmentation models trained on PIPE data

### Future Enhancements

- Expand training dataset with more diverse examples
- Improve hyperparameter optimization
- Explore alternative architectures for better semantic alignment
- Investigate mask-based approaches as complementary method

## Citation

If you use this work, please cite:

```bibtex
@misc{tsach2024visually,
  title={Visually Guided Object Insertion Into Image},
  author={Adi Tsach and Aya Spira},
  year={2024},
  note={Supervised by Noam Rotstein and Roy Gantz}
}
```

## License

This project is licensed under the Apache License 2.0 (see LICENSE file for details).

## Acknowledgments

- Based on [InstructPix2Pix](https://github.com/timbrooks/instruct-pix2pix) by Tim Brooks et al.
- Dataset derived from [PIPE](https://huggingface.co/datasets/paint-by-inpaint/PIPE) dataset
- Uses [CLIP](https://github.com/openai/CLIP) by OpenAI for image encoding
- Built with [Diffusers](https://github.com/huggingface/diffusers) library
