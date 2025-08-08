# YOLO Conversion Script

This script processes images and their associated JSON files, filters them based on specified criteria, and creates a YOLO-compatible dataset structure with train/val/test splits.

## Features

- **Recursive Processing**: Supports recursive subdirectories in both input and JSON directories
- **JSON-based Filtering**: Filters images based on JSON criteria (e.g., `correct_image`, `photorealistic`)
- **YOLO Dataset Structure**: Creates train/val/test splits with proper directory structure
- **Class Depth Control**: Configurable class name extraction based on directory depth
- **Clean Image Prioritization**: Ensures minimum percentages of clean images in validation and test sets
- **Image Renaming**: Automatically renames images to `classname_######.jpg` format
- **Comprehensive Statistics**: Detailed reporting of excluded images, non-clean images, and their distribution
- **Comprehensive Logging**: Provides detailed progress and statistics
- **Error Handling**: Gracefully handles missing files, invalid JSON, and other errors

## Usage

```bash
python convert_to_yolo.py --input-dir /path/to/images --json-dir /path/to/json --output-dir /path/to/output --test-percent 10 --val-percent 10 --remove_criteria correct_image photorealistic --test-criteria white_background no_watermarks_blemishes --class-depth 2
```

### Arguments

- `--input-dir`: Directory containing input images (supports recursive subdirectories)
- `--json-dir`: Directory containing JSON annotation files (supports recursive subdirectories)
- `--output-dir`: Output directory for YOLO dataset structure
- `--test-percent`: Percentage of images for test set split (default: 10.0)
- `--val-percent`: Percentage of images for validation set split (default: 10.0)
- `--test-clean-percent`: Minimum percentage of clean images in test set (default: 0.0)
- `--val-clean-percent`: Minimum percentage of clean images in validation set (default: 90.0)
- `--remove_criteria`: JSON keys to check for exclusion. Images with 'no' values for these keys will be excluded
- `--test-criteria`: JSON keys to check for test set. Images with 'no' values for these keys will be put in test set
- `--class-depth`: Depth level for class name extraction (default: 1)
- `--seed`: Random seed for reproducible splits (default: 42)

### Examples

1. **Basic usage with default splits**:
   ```bash
   python convert_to_yolo.py --input-dir ../inpainted_images --json-dir ../vlm_screened_images --output-dir ./yolo_dataset --remove_criteria correct_image photorealistic
   ```

2. **Custom split percentages**:
   ```bash
   python convert_to_yolo.py --input-dir ../inpainted_images --json-dir ../vlm_screened_images --output-dir ./yolo_dataset --test-percent 15 --val-percent 15 --remove_criteria correct_image photorealistic
   ```

3. **With test criteria and class depth**:
   ```bash
   python convert_to_yolo.py --input-dir ../inpainted_images --json-dir ../vlm_screened_images --output-dir ./yolo_dataset --test-percent 10 --val-percent 10 --remove_criteria correct_image photorealistic --test-criteria white_background no_watermarks_blemishes --class-depth 2
   ```

4. **Custom clean image percentages**:
   ```bash
   python convert_to_yolo.py --input-dir ../inpainted_images --json-dir ../vlm_screened_images --output-dir ./yolo_dataset --test-percent 10 --val-percent 10 --test-clean-percent 20 --val-clean-percent 95 --remove_criteria correct_image photorealistic
   ```

## JSON Structure

The script expects JSON files with the following structure:

```json
{
  "filename": "image_name.jpg",
  "class_name": "class_name",
  "correct_image": "yes",
  "photorealistic": "yes",
  "white_background": "yes",
  "no_watermarks_blemishes": "yes",
  "overall_result": "PASS"
}
```

## Dataset Structure

The script creates the following YOLO-compatible directory structure:

```
output_dir/
├── train/
│   ├── class1/
│   │   ├── class1_000000.jpg
│   │   ├── class1_000001.jpg
│   │   └── ...
│   └── class2/
│       ├── class2_000000.jpg
│       └── ...
├── val/
│   ├── class1/
│   │   ├── class1_000000.jpg
│   │   └── ...
│   └── class2/
│       └── ...
└── test/
    ├── class1/
    │   ├── class1_000000.jpg
    │   └── ...
    └── class2/
        └── ...
```

## How It Works

1. **File Discovery**: Recursively finds all `.jpg` images in the input directory and `.json` files in the JSON directory
2. **Mapping**: Creates a mapping between image files and their corresponding JSON files based on filename (without extension)
3. **Filtering**: Excludes images based on `--remove_criteria` and identifies test candidates based on `--test-criteria`
4. **Class Extraction**: Extracts class names based on `--class-depth` from the directory structure
5. **Dataset Splitting**: 
   - Splits clean images based on `--test-percent` and `--val-percent`
   - Ensures minimum clean image percentages in validation (`--val-clean-percent`) and test (`--test-clean-percent`) sets
   - Combines test candidates with percentage-based test images
6. **Image Processing**: Copies images to appropriate directories with renamed filenames
7. **Statistics Collection**: Tracks and reports detailed statistics about image distribution
8. **Reporting**: Provides comprehensive statistics including processed, excluded, and missing files

## Split Logic

- **Test Set**: Includes images that fail `--test-criteria` PLUS the specified percentage of clean images
- **Validation Set**: Prioritizes clean images (default 90% minimum) for better model evaluation
- **Training Set**: Remaining clean images after test and validation splits

## Class Depth Examples

- `--class-depth 1`: Uses the first directory level as class name
  - `metal/metal_can/aerosol_can/image.jpg` → class: `metal`
- `--class-depth 2`: Uses the second directory level as class name
  - `metal/metal_can/aerosol_can/image.jpg` → class: `metal_can`
- `--class-depth 3`: Uses the third directory level as class name
  - `metal/metal_can/aerosol_can/image.jpg` → class: `aerosol_can`

## Output and Statistics

The script provides comprehensive statistics and real-time progress updates:

```
Processing images from: ../inpainted_images
Processing JSON files from: ../vlm_screened_images
Output directory: ./yolo_dataset
Test split percentage: 10.0%
Validation split percentage: 10.0%
Test clean percentage: 0.0%
Validation clean percentage: 90.0%
Remove criteria: ['correct_image', 'photorealistic']
Test criteria: ['white_background', 'no_watermarks_blemishes']
Class depth: 2
--------------------------------------------------
Found 10000 image files
Found 10000 JSON files
Created mapping for 10000 JSON files
Excluding image1.jpg - failed criteria: ['correct_image', 'photorealistic']
Excluding image2.jpg - failed criteria: ['correct_image', 'photorealistic']
...

Processed 7500 valid images
Excluded 2500 images
Missing JSON: 0 images

Exclusion breakdown by criteria:
  correct_image: 1500 images
  photorealistic: 1000 images

Dataset analysis:
  Total valid images: 7500
  Clean images (PASS): 7000
  Non-clean images: 500
  Test criteria images: 500

Creating YOLO dataset structure...
Found 7000 clean images and 500 test candidates

Processing train split (5800 images)...
Processing val split (700 images)...
Processing test split (1200 images)...
==================================================
FINAL DATASET STATISTICS
==================================================
Dataset structure created in: ./yolo_dataset
Train images: 5800
Validation images: 700
Test images: 1200
Total images: 7700

Detailed split statistics:
  TRAIN split:
    Total images: 5800
    Clean images: 5800 (100.0%)
    Non-clean images: 0 (0.0%)
    Test criteria images: 0 (0.0%)
  VAL split:
    Total images: 700
    Clean images: 630 (90.0%)
    Non-clean images: 70 (10.0%)
    Test criteria images: 0 (0.0%)
  TEST split:
    Total images: 1200
    Clean images: 570 (47.5%)
    Non-clean images: 630 (52.5%)
    Test criteria images: 500 (41.7%)

Non-clean images (test criteria) distribution:
  Total non-clean images found: 500
  Distribution across splits:
    TRAIN: 0 images (0.0%)
    VAL: 0 images (0.0%)
    TEST: 500 images (100.0%)

Exclusion summary:
  Images excluded by criteria: 2500
  Images with missing JSON: 0
  Total excluded: 2500
```

### Statistics Explained

- **Exclusion breakdown**: Shows how many images were excluded by each specific criteria
- **Dataset analysis**: Provides overview of clean vs non-clean images and test criteria images
- **Detailed split statistics**: Shows the composition of each split (train/val/test) including:
  - Total images per split
  - Clean vs non-clean image counts and percentages
  - Test criteria image counts and percentages
- **Non-clean image distribution**: Shows how test criteria images are distributed across splits
- **Exclusion summary**: Final count of all excluded images

## Requirements

- Python 3.6+
- Standard library modules: `argparse`, `json`, `os`, `shutil`, `pathlib`, `random`, `collections`

## Notes

- The script preserves the exact directory structure from the input directory
- Images are copied (not moved) to preserve the original dataset
- The script handles missing JSON files gracefully and reports them
- Invalid JSON files are skipped with error messages
- Progress is reported every 100 processed images
- Validation set prioritizes clean images for better model evaluation
- Test set includes both criteria-based and percentage-based images
- Comprehensive statistics help understand dataset composition and quality 