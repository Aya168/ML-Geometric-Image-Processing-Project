# Visually Guided Object Insertion Into Image

## Project Information

- **Authors:** Adi Tsach & Aya Spira
- **Supervisors:** Noam Rotstein & Roy Gantz
- **Year:** 2024

## Abstract

This project presents a novel approach to image editing using diffusion models, where textual prompts are replaced with visual object examples. Unlike existing methods that require manual mask annotation, our approach enables intuitive object insertion guided solely by reference images.

## Key Contributions

1. **Embedding Optimizer Architecture**: A custom module that bridges CLIP image embeddings to Stable Diffusion's textual embedding space
2. **Mask-Free Object Insertion**: Eliminates the need for user-provided masks or coordinates
3. **Large-Scale Dataset**: Created a dataset of 108,093 image triplets from PIPE dataset
4. **Fine-tuned Model**: Successfully adapted InstructPix2Pix for image-based conditioning

## Technical Details

### Architecture Evolution

The project went through several iterations:
- **Initial attempts** with direct CLIP, Paint by Example, and IP-Adapter were unsuccessful
- **Final solution** combines:
  - CLIPVisionModel for image encoding
  - Custom Embedding Optimizer for dimension/sequence alignment
  - Fine-tuned UNet cross-attention layers

### Embedding Optimizer

The final architecture includes:
- Projection: 1024 → 768 dimensions
- Attention pooling: 257 → 77 sequence length
- 4-layer MLP refinement block

### Training Strategy

- Embedding Optimizer: Full learning rate
- UNet cross-attention: 0.1x learning rate (selective blocks only)
- VAE & Image Encoder: Frozen

## Results

The model successfully generates realistic object insertions without requiring masks, maintaining the quality and realism of the base InstructPix2Pix model while adding visual guidance capabilities.

## References

- Paint by Inpaint: [2404.18212] Paint by Inpaint: Learning to Add Image Objects by Removing Them First
- Paint by Example: [2211.13227] Paint by Example: Exemplar-based Image Editing with Diffusion Models
- PIPE Dataset: paint-by-inpaint/PIPE on Hugging Face

