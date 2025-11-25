#!/bin/bash

# --- 1. Help Message and Default Configuration ---

# Function to display help/usage information
show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

This script processes a directory of ShapeNet models using the ManifoldPlus executable
to create clean, watertight meshes.

OPTIONS:
  -e, --manifold-exe PATH   Path to the ManifoldPlus executable.
                            Default: "./ManifoldPlus/build/manifold"
  -i, --input-dir PATH      Root directory of a ShapeNet category (e.g., '02691156').
                            Default: "./data/02691156/02691156"
  -o, --output-dir PATH     Directory to save the cleaned .obj files.
                            Default: "./data/02691156/planes"
  -d, --depth INT           Manifold meshing depth (resolution).
                            Default: 8
  -h, --help                Display this help message and exit.
EOF
}

# Set default values for our variables
MANIFOLD_EXECUTABLE_DEFAULT="./ManifoldPlus/build/manifold"
INPUT_DIR_DEFAULT="./data/shapenet/03001627"
OUTPUT_DIR_DEFAULT="./data/baseline/chair"
DEPTH_DEFAULT=8

# --- 2. Parse Command-Line Arguments ---

# This loop processes arguments until none are left
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -e|--manifold-exe)
      MANIFOLD_EXECUTABLE="$2"
      shift # past argument
      shift # past value
      ;;
    -i|--input-dir)
      INPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--depth)
      DEPTH="$2"
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)    # unknown option
      echo "Error: Unknown option '$1'"
      show_help
      exit 1
      ;;
  esac
done

# --- 3. Set Final Variables (Use defaults if not provided) ---

# This clever syntax uses the default value if the variable is empty
MANIFOLD_EXECUTABLE="${MANIFOLD_EXECUTABLE:-$MANIFOLD_EXECUTABLE_DEFAULT}"
INPUT_DIR="${INPUT_DIR:-$INPUT_DIR_DEFAULT}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}"
DEPTH="${DEPTH:-$DEPTH_DEFAULT}"

# --- 4. Validation and Main Logic ---

echo "--- Configuration ---"
echo "Manifold Executable: $MANIFOLD_EXECUTABLE"
echo "Input Directory:     $INPUT_DIR"
echo "Output Directory:    $OUTPUT_DIR"
echo "Meshing Depth:       $DEPTH"
echo "---------------------"

# Check if the executable and input directory exist before starting
if [ ! -x "$MANIFOLD_EXECUTABLE" ]; then
    echo "Error: Manifold executable not found or not executable at: $MANIFOLD_EXECUTABLE"
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found at: $INPUT_DIR"
    exit 1
fi

# Create the output directory if it doesn't exist.
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Use 'find' to locate all 'model_normalized.obj' files recursively.
# Pipe the results to a 'while read' loop to process each file.
find "$INPUT_DIR" -type f -name "model_normalized.obj" | while read -r input_file; do

  # Extract the unique ID from the file path.
  unique_id=$(basename "$(dirname "$(dirname "$input_file")")")
  
  # Construct the full path for the new output file.
  output_file="$OUTPUT_DIR/$unique_id.obj"

  # If the output file already exists, skip to the next one.
  if [ -f "$output_file" ]; then
    echo "Skipping (already exists): $output_file"
    continue
  fi

  echo "Processing '$input_file'  ==>  '$output_file'"
  
  # Run ManifoldPlus on the file with the specified depth.
  "$MANIFOLD_EXECUTABLE" --input "$input_file" --output "$output_file" --depth "$DEPTH"
  
done

echo "---"
echo "All processing complete!"
echo "Cleaned meshes are in: $OUTPUT_DIR"