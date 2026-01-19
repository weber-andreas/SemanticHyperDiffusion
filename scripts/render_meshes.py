"""Renders 3D meshes from a folder and creates a grid image of the renders."""

import os
import sys
import glob
from pathlib import Path
import trimesh
import numpy as np
import pyrender
from PIL import Image, ImageOps

# Add the parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["PYOPENGL_PLATFORM"] = "egl"


def render_mesh_with_ground(mesh, color=[0.08, 0.15, 0.45, 1.0]):
    """
    Render with a top-down light to force shadows directly underneath.
    """
    # 1. Center and position mesh on ground
    mesh_copy = mesh.copy()
    mesh_copy.vertices -= mesh_copy.bounds.mean(axis=0)
    min_y = mesh_copy.bounds[0, 1]

    mesh_copy.vertices[:, 1] -= min_y

    # 2. Material (Clay)
    mesh_pr = pyrender.Mesh.from_trimesh(
        mesh_copy,
        material=pyrender.MetallicRoughnessMaterial(
            alphaMode="OPAQUE",
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )

    # 3. Ground Plane
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

    # 4. Scene
    scene = pyrender.Scene(
        ambient_light=[0.4, 0.4, 0.4],
        bg_color=[1.0, 1.0, 1.0, 1.0],
    )
    scene.add(mesh_pr)
    scene.add(ground_pr)

    # 5. Camera (Isometric)
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

    # =========================================================
    # 6. Lighting: TOP-DOWN (Shadows directly under)
    # =========================================================
    light_pose = np.eye(4)

    light_pos = np.array([0.0, 10.0, 0.0])  # Positions are floats

    # FIX: Initialize these as FLOAT arrays to allow division
    l_forward = np.array([0.0, -1.0, 0.0])
    l_up_temp = np.array([0.0, 0.0, 1.0])

    l_right = np.cross(l_forward, l_up_temp)
    l_right /= np.linalg.norm(l_right)  # Now works because l_right is float

    l_up = np.cross(l_right, l_forward)
    l_up /= np.linalg.norm(l_up)

    light_pose[:3, 0] = l_right
    light_pose[:3, 1] = l_up
    light_pose[:3, 2] = -l_forward
    light_pose[:3, 3] = light_pos

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(light, pose=light_pose)

    # 7. Rendering
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


def create_grid_image(image_paths, output_path, rows=2, cols=4):
    """Stitches images into a grid."""
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

    # Resize all to match the height of the smallest image
    target_h = min(img.height for img in images)
    resized_images = []
    for img in images:
        aspect = img.width / img.height
        new_w = int(target_h * aspect)
        resized_images.append(img.resize((new_w, target_h), Image.Resampling.LANCZOS))

    # Split into rows
    grid_matrix = []
    for r in range(rows):
        start = r * cols
        end = start + cols
        grid_matrix.append(resized_images[start:end])

    # Calculate Canvas Size
    row_widths = [sum(img.width for img in row) for row in grid_matrix]
    max_row_width = max(row_widths) + (cols + 1) * 20
    total_height = sum(row[0].height for row in grid_matrix if row) + (rows + 1) * 20

    # Create Canvas
    canvas = Image.new("RGBA", (max_row_width, total_height), (255, 255, 255, 255))

    current_y = 20
    for row_imgs in grid_matrix:
        if not row_imgs:
            continue

        row_w = sum(img.width for img in row_imgs)
        spacing = (max_row_width - row_w) // (len(row_imgs) + 1)

        current_x = spacing
        for img in row_imgs:
            canvas.paste(img, (current_x, current_y), img)
            current_x += img.width + spacing

        current_y += row_imgs[0].height + 20

    canvas.save(output_path)
    print(f"Grid saved to {output_path}")


def render_and_grid(mesh_folder, output_folder, rows=2, cols=4):
    os.makedirs(output_folder, exist_ok=True)

    # 1. Get all files
    all_files = sorted(glob.glob(os.path.join(mesh_folder, "*.obj")))
    # Fallback to .ply if needed
    if not all_files:
        all_files = sorted(glob.glob(os.path.join(mesh_folder, "*.ply")))

    print(f"Found {len(all_files)} files total.")

    if len(all_files) < 8:
        print("Warning: Found fewer than 8 files. Grid might look incomplete.")

    row_mapping = {
        f"Row {i+1}": all_files[i * cols : (i + 1) * cols] for i in range(rows)
    }

    final_list_for_grid = []

    for label, file_list in row_mapping.items():
        print(f"\nProcessing {label}...")

        row_images = []
        for mesh_path in file_list:
            print(f"  Rendering {os.path.basename(mesh_path)}...")
            try:
                mesh = trimesh.load(mesh_path)

                # Render (returns PIL Image)
                pil_img = render_mesh_with_ground(mesh)

                # Save individual render
                out_name = f"{Path(mesh_path).stem}_render.png"
                out_path = os.path.join(output_folder, out_name)
                pil_img.save(out_path)

                row_images.append(out_path)
            except Exception as e:
                print(f"  Error: {e}")

        final_list_for_grid.extend(row_images)

        # Fill row with placeholders if fewer than 4 items found
        while len(row_images) < cols:
            final_list_for_grid.append("empty_placeholder")

    # 3. Generate Final Grid
    if final_list_for_grid:
        grid_out = os.path.join(output_folder, "final_shape_grid.png")
        create_grid_image(final_list_for_grid, grid_out, rows=rows, cols=cols)


if __name__ == "__main__":
    # Adjust paths here
    mesh_folder = "gen_meshes/dummy-9q2wftm0"
    output_folder = "visualizations/rendered_meshes"

    render_and_grid(mesh_folder, output_folder, rows=2, cols=5)
