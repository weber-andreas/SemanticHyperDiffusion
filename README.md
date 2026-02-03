# SemanticHyperDiffusion

**Semantic Decomposition of MLP Weight-Space for 3D Novel Shape Generation**

*A Student Project by Andreas Weber and Thomas Linder @ Technical University of Munich (TUM)*

Based on the official implementation of ["HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion" (ICCV 2023)](https://arxiv.org/abs/2303.17015).

## Method Overview

We extend the HyperDiffusion framework by introducing a **semantically enriched weight-space diffusion process**. Instead of treating MLP parameters as a single undifferentiated vector, we explicitly decompose and model semantically related subsets of weights (e.g., wings, body, engine).

Key features of our approach:
*   **Semantic Overfitting:** Utilizes a Mixture of Experts (MoE) architecture where specific expert networks are trained on specific semantic parts of the 3D geometry.
*   **Structured Diffusion:** Performs diffusion on a structured weight space, enabling the model to learn correlations between geometry and part-level semantics.
*   **Downstream Applications:** Enables novel capabilities such as part-wise shape generation and part-wise interpolation.

## Installation & Setup

1. **Environment:**
   ```sh
   conda env create -f hyperdiffusion_env.yaml
   conda activate hyper-diffusion
   export PYTHONPATH="."
   ```

2. **Preprocessing:**
   We utilize the **ShapeNetPart dataset**. We use [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus) to convert 3D triangle meshes into watertight manifold meshes before processing.

## Usage

Our pipeline consists of two main stages: Semantic Overfitting and Weight-Space Diffusion.

### 1. Semantic Overfitting
Instead of standard overfitting, we optimize MLPs using a semantic decomposition strategy (MoE).

To overfit shapes with semantic experts (e.g., for airplanes):
```commandline
python src/mlp_decomposition/overfit_mlp.py --config-name=overfit_plane_equal
```

### 2. Diffusion Training & Sampling
Once the semantically decomposed weights are generated, they are flattened and used to train the diffusion transformer.

To train the diffusion model using the MoE configuration:
```commandline
python src/main.py --config-name=train_plane_moe
```

> **Note on Training Batches:** 
> Due to filtering bad shapes (e.g., 488 filtered from 3237 total planes), the effective dataset size is approx 2749 samples. With a batch size of 8, this results in roughly 344 batches per epoch.

## Downstream Applications

Our structured representation enables manipulation of specific object parts directly in weight space.

### Part-wise Generation (Hybrid Shapes)
Generate new shapes by stitching semantic parts from different "parent" latent vectors.
```commandline
python scripts/generate_hybrid_shapes.py --ckpt <path_to_diffusion_ckpt> --parts_A engine --parts_B body
```

### Part Interpolation
Interpolate the geometry of a specific semantic part (e.g., wings) between two shapes while keeping other parts fixed.
```commandline
python scripts/render_part_interpolation.py --checkpoint_path1 <path_to_mlp_1> --checkpoint_path2 <path_to_mlp_2> --part_name wing
```

## Evaluation Metrics

We evaluate generation quality using the following metrics, primarily based on Chamfer Distance (CD):

*   **Minimum Matching Distance (MMD):** Measures **Quality/Fidelity**. Lower scores indicate generated shapes are closer to the real distribution.
*   **Coverage (COV):** Measures **Diversity**. Higher scores indicate the model covers more of the real distribution modes.
*   **1-Nearest Neighbor Accuracy (1-NNA):** Measures **Distribution Similarity**. An accuracy of ~50% is ideal (indistinguishable from real). >50% implies low quality, <50% implies overfitting.

## Code Structure

This repository builds upon the original codebase. Key additions for the Semantic project include:

*   **`src/mlp_decomposition/`**: Contains the core logic for the Mixture of Experts (MoE) architecture and semantic loss functions.
*   **`configs/`**: Contains specific configs for semantic overfitting (`overfit_plane_equal.yaml`) and diffusion (`train_plane_moe.yaml`).
*   **`scripts/`**: Contains downstream application scripts and analysis tools.

## Acknowledgments

This project is based on the work of Erkoç et al. (HyperDiffusion). We thank the original authors for open-sourcing their code.

**Citation for the original work:**
```bibtex
@misc{erkoc2023hyperdiffusion,
  title={HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion},
  author={Ziya Erkoç and Fangchang Ma and Qi Shan and Matthias Nießner and Angela Dai},
  year={2023},
  eprint={2303.17015},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
