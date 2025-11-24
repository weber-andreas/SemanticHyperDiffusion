# SemanticHyperDiffusion
Semantic Decomposition of MLP weights to improve 3D novel shape generation leveraging diffusion models.

Make sure to execute `export PYTHONPATH="."` before running the code.

## Training Details

### Computation of Batch Size
1. Total Training Shapes: 3237 (from train_split.lst)
2. Filtered Shapes: 488 shapes are removed because filter_bad: True (listed in plane_problematic_shapes.txt).
3. Effective Dataset Size: 3237 - 488 = 2749 samples.
4. Batches per Epoch: ceil(2749 / 8) = 344.



## Metrics
**Chamfer Distance**:
<br>All metrics are based on the Chamfer Distance (CD).
It calculates the average distance from every point in one point cloud to its nearest neighbor in the other point cloud, and vice versa. 

**Earth Mover's Distance (EMD)**:
<br>It measures the minimum amount of work needed to transform one point cloud into another. 

**Pairwise Distances (pairwise_EMD_CD)**:
<br>It first calculates the Chamfer Distance (CD) and Earth Mover's Distance (EMD) between every generated sample and every reference (real) point cloud.

**Minimum Matching Distance (MMD-CD)**:
<br>What it is: For each reference (real) shape, it finds the closest generated shape and averages these distances.
Goal: Measures Quality/Fidelity. A lower score means the generated shapes are very similar to the real shapes.

**Coverage (COV-CD)**:
<br>What it is: The fraction of reference (real) shapes that are the "nearest neighbor" to at least one generated shape.
Goal: Measures Diversity. A higher score means the model is generating a wide variety of shapes that cover the real distribution, rather than collapsing to a few modes.

**1-Nearest Neighbor Accuracy (1-NN-CD)**:
<br>What it is: A classifier tries to distinguish between real and generated point clouds based on their nearest neighbor in the combined set.
Goal: Measures Distribution Similarity.
50% Accuracy: Ideal. The generated shapes are indistinguishable from real ones.

\> 50% Accuracy: The classifier can easily tell them apart (bad).
<br>\< 50% Accuracy: Indicates overfitting (the model is copying the training set).