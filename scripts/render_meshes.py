import os
import sys
import glob
from pathlib import Path
import trimesh
import numpy as np
import pyrender
from PIL import Image, ImageOps
import re
from scipy.spatial import cKDTree

# Add the parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["PYOPENGL_PLATFORM"] = "egl"


def clean_mesh(mesh):
    """
    Keep the main object and any parts visually close to it.
    Uses KDTree for fast proximity checks.
    """
    # Split the mesh into connected components
    components = mesh.split(only_watertight=False)

    if len(components) < 2:
        return mesh

    # Sort by vertex count (descending) to find the "Main Body"
    # Convert generator to list first
    components = list(components)
    components.sort(key=lambda m: len(m.vertices), reverse=True)
    main_body = components[0]

    # 1. Build a KDTree for the main body vertices (One-time cost, very fast)
    main_tree = cKDTree(main_body.vertices)

    # Calculate threshold (e.g., 5% of the diagonal size)
    # 0.1% (0.001) was likely too strict for generated meshes
    diag = np.linalg.norm(main_body.bounds[1] - main_body.bounds[0])
    threshold = diag * 0.05

    kept_components = [main_body]

    # Pre-calculate main body bounds for fast rejection
    main_bounds = main_body.bounds
    # Expand bounds by threshold to creating a "safe zone"
    expanded_bounds = [main_bounds[0] - threshold, main_bounds[1] + threshold]

    for comp in components[1:]:
        # 2. Fast Bounding Box Check (Discard obviously far objects)
        # If the component's bounds don't overlap with the expanded main bounds, drop it.
        if np.any(comp.bounds[1] < expanded_bounds[0]) or np.any(
            comp.bounds[0] > expanded_bounds[1]
        ):
            continue

        # 3. KDTree Distance Check
        # We only need to know if *any* point of the component is close to the main body.
        # Subsample if the component is huge (e.g., max 100 points)
        samples = comp.vertices
        if len(samples) > 100:
            # Deterministic slicing is faster than random choice
            samples = samples[:: len(samples) // 100]

        # Query the tree: distance to the nearest neighbor in main_body
        # k=1 returns (distances, indices)
        dists, _ = main_tree.query(samples, k=1, workers=1)

        # If the minimum distance is within threshold, keep the component
        if np.min(dists) < threshold:
            kept_components.append(comp)

    # Merge kept parts
    return trimesh.util.concatenate(kept_components)


def render_mesh_with_ground(mesh, color=[0.08, 0.15, 0.45, 1.0], skip_cleanup=False):
    """
    Render with a top-down light to force shadows directly underneath.
    Auto-scales mesh to ensure consistent size.

    Args:
        mesh: trimesh object to render
        color: RGBA color for the mesh
        skip_cleanup: if True, skip the mesh cleaning step (useful for neural implicit meshes)
    """
    # 0. CLEANUP: Remove outliers (Keep only main object)
    if skip_cleanup:
        mesh_copy = mesh.copy()
    else:
        mesh_copy = clean_mesh(mesh.copy())

    mesh_copy.merge_vertices()
    # flip normals to face outward
    mesh_copy.fix_normals()

    # 1. Center the mesh
    mesh_copy.vertices -= mesh_copy.bounds.mean(axis=0)

    # 2. NORMALIZE SCALE
    # Scale the mesh so its longest side is exactly 2.0 units.
    max_extent = np.max(mesh_copy.extents)
    if max_extent > 0:
        scale_factor = 2.25 / max_extent
        mesh_copy.apply_scale(scale_factor)

    # 3. Re-Center and Floor Align after scaling
    mesh_copy.vertices -= mesh_copy.bounds.mean(axis=0)
    min_y = mesh_copy.bounds[0, 1]
    mesh_copy.vertices[:, 1] -= min_y  # Place feet on ground

    # [Rest of your rendering code remains exactly the same]
    # Material (Clay)
    mesh_pr = pyrender.Mesh.from_trimesh(
        mesh_copy,
        material=pyrender.MetallicRoughnessMaterial(
            alphaMode="OPAQUE",
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
        ),
    )

    # Ground Plane
    ground_size = 50.0
    ground_mesh = trimesh.creation.box(extents=[ground_size, 0.01, ground_size])
    ground_mesh.apply_translation([0, -0.005, 0])
    ground_pr = pyrender.Mesh.from_trimesh(
        ground_mesh,
        material=pyrender.MetallicRoughnessMaterial(
            alphaMode="OPAQUE",
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )

    # Scene
    scene = pyrender.Scene(
        ambient_light=[0.4, 0.4, 0.4],
        bg_color=[1.0, 1.0, 1.0, 1.0],
    )
    scene.add(mesh_pr)
    scene.add(ground_pr)

    # Camera (Isometric)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    eye = np.array([2.0, 1.5, -2.0])
    target = np.array([0.0, 0.3, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)

    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = new_up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = eye
    scene.add(camera, pose=camera_pose)

    # Lighting (Top Down)
    light_pose = np.eye(4)
    light_pos = np.array([0.0, 10.0, 0.0])
    l_forward = np.array([0.0, -1.0, 0.0])
    l_up_temp = np.array([0.0, 0.0, 1.0])

    l_right = np.cross(l_forward, l_up_temp)
    l_right /= np.linalg.norm(l_right)
    l_up = np.cross(l_right, l_forward)
    l_up /= np.linalg.norm(l_up)

    light_pose[:3, 0] = l_right
    light_pose[:3, 1] = l_up
    light_pose[:3, 2] = -l_forward
    light_pose[:3, 3] = light_pos

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(light, pose=light_pose)

    # Rendering
    r = pyrender.OffscreenRenderer(1600, 1600)
    color_img, _ = r.render(
        scene,
        flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.ALL_SOLID,
    )

    img_pil = Image.fromarray(color_img)
    return img_pil


def crop_whitespace(img_pil):
    """Auto-crops the white borders from an image."""
    bbox = ImageOps.invert(img_pil.convert("RGB")).getbbox()
    if bbox:
        return img_pil.crop(bbox)
    return img_pil


def create_grid_image(
    image_paths,
    output_path,
    rows=2,
    cols=4,
    pad=6,
    row_gap=6,
    col_gap=6,
    square_tiles=True,
):
    """Stitches images into a grid with controllable spacing.

    Args:
        image_paths: list of image file paths to include in the grid
        output_path: where to save the resulting grid image
        rows: number of rows in the grid
        cols: number of columns in the grid
        pad: outer margin around the entire grid (pixels)
        row_gap: vertical gap between rows (pixels)
        col_gap: horizontal gap between images in a row (pixels)
        square_tiles: if True, place each image on a square tile; if False, keep tight image bounds
    """
    images = []
    for p in image_paths:
        if os.path.exists(p):
            img = Image.open(p).convert("RGBA")
            img = crop_whitespace(img)
            images.append(img)
        else:
            images.append(Image.new("RGBA", (100, 100), (255, 255, 255, 0)))

    if not images:
        return

    heights = [img.height for img in images]
    widths = [img.width for img in images]
    target_size = int(max(np.percentile(heights, 90), np.percentile(widths, 90)))
    target_size = max(target_size, 800)

    resized_images = []
    for img in images:
        if img.width > target_size or img.height > target_size:
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

        if square_tiles:
            canvas = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 0))
            x_offset = (target_size - img.width) // 2
            y_offset = (target_size - img.height) // 2
            canvas.paste(img, (x_offset, y_offset), img)
            resized_images.append(canvas)
        else:
            # Keep tight bounds; no extra padding
            resized_images.append(img)

    grid_matrix = []
    for r in range(rows):
        start = r * cols
        end = start + cols
        grid_matrix.append(resized_images[start:end])

    # Compute canvas size using explicit gaps and outer padding
    # Compute width per row using the actual number of images in that row
    row_widths = [
        (sum(img.width for img in row) + (len(row) - 1) * col_gap) if row else 0
        for row in grid_matrix
    ]
    max_row_width = (max(row_widths) if row_widths else 0) + 2 * pad
    # Compute row heights using max tile height when not square
    row_heights = [
        (max(img.height for img in row) if row else 0) for row in grid_matrix
    ]
    total_height = sum(row_heights) + (rows - 1) * row_gap + 2 * pad

    canvas = Image.new("RGBA", (max_row_width, total_height), (255, 255, 255, 255))

    current_y = pad
    for row_imgs in grid_matrix:
        if not row_imgs:
            continue

        current_x = pad
        # Paste images top-aligned within the row
        max_h = max(img.height for img in row_imgs)
        for img in row_imgs:
            canvas.paste(img, (current_x, current_y), img)
            current_x += img.width + col_gap

        current_y += max_h + row_gap

    canvas.save(output_path)
    print(f"Grid saved to {output_path}")


def render_meshes(mesh_folder, output_folder, mesh_list=None):
    os.makedirs(output_folder, exist_ok=True)

    def extract_number(filename):
        match = re.search(r"\d+", os.path.basename(filename))
        return int(match.group()) if match else 0

    if mesh_list is not None:
        meshes_to_render = mesh_list
    else:
        all_obj = sorted(
            glob.glob(os.path.join(mesh_folder, "*.obj")), key=extract_number
        )
        all_ply = sorted(
            glob.glob(os.path.join(mesh_folder, "*.ply")), key=extract_number
        )
        meshes_to_render = all_obj + all_ply

    print(f"Found {len(meshes_to_render)} mesh files to render.")

    rendered_images = []

    for mesh_ref in meshes_to_render:
        if isinstance(mesh_ref, int):
            continue
        elif os.path.exists(mesh_ref):
            mesh_path = mesh_ref
        elif os.path.isabs(mesh_ref):
            mesh_path = mesh_ref
        else:
            mesh_path = os.path.join(mesh_folder, mesh_ref)

        if not os.path.exists(mesh_path):
            print(f"  Warning: {mesh_ref} not found, skipping...")
            continue

        print(f"  Rendering {os.path.basename(mesh_path)}...")
        try:
            mesh = trimesh.load(mesh_path)
            pil_img = render_mesh_with_ground(mesh)

            out_name = f"{Path(mesh_path).stem}_render.png"
            out_path = os.path.join(output_folder, out_name)
            pil_img.save(out_path)

            rendered_images.append(out_path)
            print(f"    Saved: {out_path}")
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nSuccessfully rendered {len(rendered_images)} meshes.")
    return rendered_images


def create_grid(
    image_paths,
    output_path,
    rows=2,
    cols=4,
    pad=6,
    row_gap=6,
    col_gap=6,
    square_tiles=True,
):
    create_grid_image(
        image_paths,
        output_path,
        rows=rows,
        cols=cols,
        pad=pad,
        row_gap=row_gap,
        col_gap=col_gap,
        square_tiles=square_tiles,
    )


if __name__ == "__main__":
    mesh_folder = "gen_meshes/semanticHD_filtered"
    output_folder = "visualizations/rendered_meshes/semantic_hd_plane"

    specific_meshes = [
        "mesh_0.obj",
        "mesh_1.obj",
        "mesh_2.obj",
        "mesh_3.obj",
        "mesh_12.obj",
        "mesh_5.obj",
        "mesh_6.obj",
        "mesh_7.obj",
        "mesh_13.obj",
        "mesh_9.obj",
        "mesh_10.obj",
        "mesh_11.obj",
    ]

    images = render_meshes(mesh_folder, output_folder, mesh_list=specific_meshes)
    rows = 3
    cols = 4

    create_grid(
        images,
        os.path.join(output_folder, "plane_shapes_grid.png"),
        rows=rows,
        cols=cols,
        pad=2,
        row_gap=-5,
        col_gap=5,
        square_tiles=False,
    )
