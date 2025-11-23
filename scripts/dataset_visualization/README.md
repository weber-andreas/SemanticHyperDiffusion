# ShapeNetPart Visualization

## 2D/3D Single Object Visualization

**Visualization by file ID**
```sh
python scripts/dataset_visualization/viz_shapenetpart.py --single_object --file_id 1a888c2c86248bbcf2b0736dd4d8afe0
```

**Visualization by object index**
```sh
python scripts/dataset_visualization/viz_shapenetpart.py --single_object --object_index 3 --visualize_2d
```

## Matrix Visualization
```sh
python scripts/dataset_visualization/viz_shapenetpart.py --category_matrix --categories "Airplane Chair"
```
