"""Render 3D meshes from two MLPs with interpolated parts, showing full objects side-by-side."""

import os
import re
import torch
import configargparse
import trimesh
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import sys

# Add the parent directory to path to import from src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "external/"))
sys.path.append(os.path.join(ROOT_DIR, "external/siren/"))

from external.siren import sdf_meshing
from src.mlp_decomposition.test_mlp import SDFDecoder
from scripts.render_meshes import render_mesh_with_ground

os.environ["PYOPENGL_PLATFORM"] = "egl"
DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def extract_file_id(checkpoint_path):
    """Extract the file ID from checkpoint filename using regex pattern occ_<ID>_model_final."""
    filename = Path(checkpoint_path).stem
    match = re.search(r"occ_([a-f0-9]+)_model_final", filename)
    if match:
        return match.group(1)
    return filename


def interpolate_part_parameters(model1, model2, part_name, alpha=0.5):
    """Interpolate a specific part between two models while keeping all other parts from model1."""
    # Create a copy of model1 to start with
    interpolated_model = type(model1.model)(model1.model.registry).to(DEVICE)
    interpolated_model.load_state_dict(model1.model.state_dict())

    # Check if the part exists in both models
    if part_name not in model1.model.parts:
        raise ValueError(
            f"Part '{part_name}' not found in model1. Available parts: {list(model1.model.parts.keys())}"
        )
    if part_name not in model2.model.parts:
        raise ValueError(
            f"Part '{part_name}' not found in model2. Available parts: {list(model2.model.parts.keys())}"
        )

    # Interpolate the parameters of the specified part
    part1_state = model1.model.parts[part_name].state_dict()
    part2_state = model2.model.parts[part_name].state_dict()

    interpolated_part_state = {}
    for key in part1_state.keys():
        # Linear interpolation: alpha * param1 + (1 - alpha) * param2
        interpolated_part_state[key] = (
            alpha * part1_state[key] + (1 - alpha) * part2_state[key]
        )

    # Load the interpolated parameters into the new model
    interpolated_model.parts[part_name].load_state_dict(interpolated_part_state)

    return interpolated_model


def mesh_from_model(model, output_dir, resolution=256, name="mesh"):
    """Generate a complete mesh from an MLP model using marching cubes."""
    os.makedirs(os.path.join(output_dir, "ply_files"), exist_ok=True)
    mesh_path = os.path.join(output_dir, "ply_files", name)

    print(f"Generating mesh '{name}' at resolution {resolution}...")
    vertices, faces, sdf = sdf_meshing.create_mesh(
        model,
        mesh_path,
        N=resolution,
        level=0,
        device=DEVICE,
    )

    mesh_file = f"{mesh_path}.ply"
    print(f"  Saved mesh to {mesh_file}")
    return mesh_file


def render_and_save_mesh(mesh_path, output_path, render_scale=1.0):
    """Load a mesh, render it, and save as an image."""
    mesh = trimesh.load(mesh_path)
    rendered_img = render_mesh_with_ground(
        mesh, skip_cleanup=False, scale_multiplier=render_scale
    )
    rendered_img.save(output_path)
    print(f"  Saved render to {output_path}")
    return rendered_img


from PIL import Image, ImageDraw, ImageFont


from PIL import Image, ImageDraw, ImageFont
import re


def create_side_by_side_image(
    images, output_path, labels=None, padding=40, font_size=80
):
    """
    Creates a side-by-side grid with centered headers.
    Automatically forces 'alpha' parameters and long IDs to new lines.
    """
    if not images:
        print("No images to process.")
        return

    # 1. Load a high-quality TrueType font
    font_names = ["arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf", "tahoma.ttf"]
    font = None
    for name in font_names:
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # 2. Helper: Auto-Format Label (Insert explicit line breaks)
    def format_label(text):
        if not text:
            return ""
        text = str(text)

        # A. Force break before 'alpha='
        text = text.replace(" alpha=", "\nalpha=")

        # B. Force break before long IDs (words > 15 chars)
        # Regex looks for a space followed by 15+ alphanumeric characters
        # and replaces the space with a newline.
        text = re.sub(r" ([a-zA-Z0-9]{15,})", r"\n\1", text)

        return text

    # 3. Helper: Wrap text (Respecting explicit newlines)
    def wrap_text(text, font, max_width, draw_ctx):
        # Split by existing newlines first (from our format_label step)
        paragraphs = text.split("\n")
        final_lines = []

        for paragraph in paragraphs:
            words = paragraph.split()
            current_line = []

            for word in words:
                # Test width of adding this word
                test_line = " ".join(current_line + [word])
                bbox = draw_ctx.textbbox((0, 0), test_line, font=font)
                width = bbox[2] - bbox[0]

                if width <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        final_lines.append(" ".join(current_line))
                    current_line = [word]

            # Flush the last line of this paragraph
            if current_line:
                final_lines.append(" ".join(current_line))

        return final_lines

    # 4. Pre-calculate layout
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    sample_bbox = dummy_draw.textbbox((0, 0), "Ay", font=font)
    line_height = (sample_bbox[3] - sample_bbox[1]) * 1.2

    processed_labels = []
    max_header_height = 0

    for i, img in enumerate(images):
        if labels and i < len(labels) and labels[i]:
            # Apply formatting then wrapping
            formatted_text = format_label(labels[i])
            lines = wrap_text(formatted_text, font, img.width - 20, dummy_draw)

            header_h = len(lines) * line_height + 40
            max_header_height = max(max_header_height, header_h)
            processed_labels.append(lines)
        else:
            processed_labels.append([])

    # 5. Create Canvas
    img_widths = [img.width for img in images]
    img_heights = [img.height for img in images]
    total_width = sum(img_widths) + padding * (len(images) - 1)
    max_img_height = max(img_heights)

    total_height = int(max_header_height + max_img_height + 20)

    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    # 6. Draw Content
    current_x = 0
    for i, img in enumerate(images):
        # A. Paste Image
        img_y = int(max_header_height + (max_img_height - img.height) // 2)
        canvas.paste(img, (current_x, img_y))

        # B. Draw Centered Text
        lines = processed_labels[i]
        if lines:
            text_block_h = len(lines) * line_height
            cursor_y = (max_header_height - text_block_h) / 2
            center_x = current_x + img.width / 2

            for line in lines:
                draw.text(
                    (center_x, cursor_y),
                    line,
                    font=font,
                    fill="black",
                    anchor="mt",
                    align="center",
                )
                cursor_y += line_height

        current_x += img.width + padding

    canvas.save(output_path)
    print(f"Saved side-by-side image to {output_path}")
    return canvas


def render_part_interpolation(
    checkpoint_path1,
    checkpoint_path2,
    mlp_config,
    part_name,
    output_dir,
    alpha=0.2,
    resolution=256,
    output_type="occ",
    render_scale=1.0,
):
    """Render two MLPs and their interpolated version with a specific part interpolated."""
    # Load config
    cfg = load_config(mlp_config)
    print(f"Loaded config from {mlp_config}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load both models
    print("Loading Model 1")
    model1 = SDFDecoder(checkpoint_path1, DEVICE, cfg, output_type=output_type)

    print("Loading Model 2")
    model2 = SDFDecoder(checkpoint_path2, DEVICE, cfg, output_type=output_type)

    # Check available parts
    if hasattr(model1.model, "parts"):
        available_parts = list(model1.model.parts.keys())
        if part_name not in available_parts:
            print(
                f"Warning: Part '{part_name}' not found. Using first available part: {available_parts[0]}"
            )
            part_name = available_parts[0]
    else:
        raise ValueError(
            "Model does not have a 'parts' attribute. Make sure you're using a composite or MoE model."
        )

    # Ensure models are in eval mode for inference
    model1.model.eval()
    model2.model.eval()

    # Create interpolated model
    interpolated_model_module = interpolate_part_parameters(
        model1, model2, part_name, alpha
    )

    # Create a wrapper for the interpolated model that matches SDFDecoder behavior
    class InterpolatedDecoder(torch.nn.Module):
        """Wrapper that renders the COMPLETE object with all parts combined.
        Mimics the behavior of SDFDecoder for consistency.

        For MLPMoE models, this returns the max occupancy across all parts.
        For MLPComposite models, this returns the min SDF across all parts.
        """

        def __init__(self, model, device):
            super().__init__()
            self.model = model.to(device)
            self.model.eval()  # Set to eval mode for inference
            self.device = device

        def forward(self, coords):
            # Input: coords of shape [N, 3]
            # MLPMoE/MLPComposite expect [B, N, 3]
            if coords.dim() == 2:
                # Add batch dimension if needed
                coords = coords.unsqueeze(0)

            model_in = {"coords": coords}

            # Forward pass through all parts
            model_out = self.model(model_in)

            # Return the combined output from ALL parts
            # For MLPMoE: max occupancy across all experts
            # For MLPComposite: min SDF across all parts
            return model_out["model_out"]

    interpolated_model = InterpolatedDecoder(interpolated_model_module, DEVICE)

    # Wrap in SDFDecoder-like wrapper for mesh generation
    class SDFDecoderWrapper(torch.nn.Module):
        """Wrapper to make InterpolatedDecoder compatible with sdf_meshing.create_mesh"""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, coords):
            # sdf_meshing.create_mesh calls forward with [N, 3] tensor
            return self.model(coords)

    interpolated_model_for_meshing = SDFDecoderWrapper(interpolated_model)

    # Generate meshes
    name1 = Path(checkpoint_path1).stem
    name2 = Path(checkpoint_path2).stem

    mesh_path1 = mesh_from_model(model1, output_dir, resolution, f"{name1}_mesh")
    mesh_path2 = mesh_from_model(model2, output_dir, resolution, f"{name2}_mesh")
    mesh_path_interp = mesh_from_model(
        interpolated_model_for_meshing,
        output_dir,
        resolution,
        f"interpolated_{part_name}_alpha{alpha:.2f}_mesh",
    )

    # Render meshes
    img1 = render_and_save_mesh(
        mesh_path1,
        os.path.join(output_dir, f"{name1}_render.png"),
        render_scale=render_scale,
    )
    img2 = render_and_save_mesh(
        mesh_path2,
        os.path.join(output_dir, f"{name2}_render.png"),
        render_scale=render_scale,
    )
    img_interp = render_and_save_mesh(
        mesh_path_interp,
        os.path.join(
            output_dir, f"interpolated_{part_name}_alpha{alpha:.2f}_render.png"
        ),
        render_scale=render_scale,
    )

    # Create side-by-side comparison
    id1 = extract_file_id(checkpoint_path1)
    id2 = extract_file_id(checkpoint_path2)

    labels = [
        f"Model 1 {id1}",
        f"Interpolated {part_name} alpha={alpha:.2f}",
        f"Model 2 {id2}",
    ]
    combined_path = os.path.join(
        output_dir, f"comparison_{part_name}_alpha{alpha:.2f}.png"
    )
    create_side_by_side_image(
        [img1, img_interp, img2], combined_path, labels=labels, padding=30, font_size=70
    )

    print(f"Output saved to {output_dir}")


def parse_arguments():
    """Parse and return command-line arguments."""
    p = configargparse.ArgumentParser(
        description="Render part interpolation between two MLP checkpoints."
    )
    p.add_argument("-c", "--config_filepath", required=False, is_config_file=True)

    # Required paths
    p.add_argument(
        "--checkpoint_path1",
        type=str,
        required=True,
        help="Path to the first MLP checkpoint file.",
    )
    p.add_argument(
        "--checkpoint_path2",
        type=str,
        required=True,
        help="Path to the second MLP checkpoint file.",
    )
    p.add_argument(
        "--mlp_config",
        type=str,
        required=True,
        help="Path to the MLP config YAML file describing architecture.",
    )
    p.add_argument(
        "--part_name",
        type=str,
        required=True,
        help="Name of the part to interpolate (e.g., 'wing', 'tail', 'body', 'engine').",
    )

    # Optional parameters
    p.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations/part_interpolation",
        help="Output directory for generated meshes and renders.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Interpolation factor (0 = model1, 1 = model2, 0.5 = midpoint).",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution for marching cubes mesh extraction.",
    )
    p.add_argument(
        "--output_type",
        type=str,
        default="occ",
        choices=["occ", "sdf", "logits"],
        help="Output type of the MLP model.",
    )

    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Additional scale multiplier applied during rendering (e.g., 0.7 to shrink).",
    )

    return p.parse_args()


def main():
    """Execute the part interpolation and rendering pipeline."""
    args = parse_arguments()

    # Validate paths
    if not os.path.exists(args.checkpoint_path1):
        print(f"Error: Checkpoint 1 not found: {args.checkpoint_path1}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint_path2):
        print(f"Error: Checkpoint 2 not found: {args.checkpoint_path2}")
        sys.exit(1)
    if not os.path.exists(args.mlp_config):
        print(f"Error: Config file not found: {args.mlp_config}")
        sys.exit(1)

    # Run interpolation and rendering
    render_part_interpolation(
        args.checkpoint_path1,
        args.checkpoint_path2,
        args.mlp_config,
        args.part_name,
        args.output_dir,
        alpha=args.alpha,
        resolution=args.resolution,
        output_type=args.output_type,
        render_scale=args.scale,
    )


if __name__ == "__main__":
    main()

# Plane
"""
python scripts/render_part_interpolation.py \
  --checkpoint_path1 mlp_weights/overfit_plane_new_loss/occ_2af93e42ceca0ff7efe7c6556ea140b4_model_final.pth \
  --checkpoint_path2 mlp_weights/overfit_plane_new_loss/occ_1d7eb22189100710ca8607f540cc62ba_model_final.pth \
  --mlp_config configs/overfitting_configs/overfit_plane_equal.yaml \
  --part_name body \
  --alpha 0.1 \
  --resolution 256 \
  --output_dir visualizations/part_interpolation/03/body_interpolation
  """


# Chair
"""
python scripts/render_part_interpolation.py \
  --checkpoint_path1 logs/overfit_chair_vmap/occ_1b5fc54e45c8768490ad276cd2af3a4_model_final.pth\
  --checkpoint_path2 logs/overfit_chair_vmap/occ_19cbb7fd3ba9324489a921f93b6641da_model_final.pth \
  --mlp_config configs/overfitting_configs/overfit_chair_equal.yaml \
  --scale 0.7 \
  --part_name seat \
  --alpha 0.2 \
  --resolution 256 \
  --output_dir visualizations/part_interpolation/chair/02/seat
  """
