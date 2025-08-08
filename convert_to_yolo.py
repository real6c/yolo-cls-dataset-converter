#!/usr/bin/env python3
"""
YOLO Conversion Script

This script processes images and their associated JSON files, filters them based on specified criteria,
and creates a YOLO-compatible dataset structure with train/val/test splits.

Usage:
    python convert_to_yolo.py --input-dir /path/to/images --json-dir /path/to/json --output-dir /path/to/output --test-percent 10 --val-percent 10 --remove_criteria correct_image photorealistic --test-criteria white_background no_watermarks_blemishes --class-depth 2
"""

import argparse
import json
import os
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict


def find_files_recursively(directory: Path, extension: str) -> List[Path]:
    """
    Recursively find all files with the specified extension in the given directory.
    
    Args:
        directory: Root directory to search
        extension: File extension to search for (e.g., '.jpg', '.json')
    
    Returns:
        List of Path objects for matching files
    """
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(extension.lower()):
                files.append(Path(root) / filename)
    return files


def should_exclude_image(json_data: dict, remove_criteria: List[str]) -> bool:
    """
    Check if an image should be excluded based on the specified criteria.
    
    Args:
        json_data: Dictionary containing the JSON data
        remove_criteria: List of criteria keys to check
    
    Returns:
        True if the image should be excluded, False otherwise
    """
    for criterion in remove_criteria:
        if criterion in json_data:
            if json_data[criterion].lower() == 'no':
                return True
    return False


def should_put_in_test(json_data: dict, test_criteria: List[str]) -> bool:
    """
    Check if an image should be put in the test set based on the specified criteria.
    
    Args:
        json_data: Dictionary containing the JSON data
        test_criteria: List of criteria keys to check
    
    Returns:
        True if the image should be put in test set, False otherwise
    """
    for criterion in test_criteria:
        if criterion in json_data:
            if json_data[criterion].lower() == 'no':
                return True
    return False


def get_class_name_from_path(image_path: Path, input_dir: Path, class_depth: int) -> str:
    """
    Extract class name from image path based on class depth.
    
    Args:
        image_path: Path to the image file
        input_dir: Base input directory
        class_depth: Depth level for class name extraction
    
    Returns:
        Class name based on the specified depth
    """
    try:
        relative_path = image_path.relative_to(input_dir)
        path_parts = relative_path.parts
        
        if class_depth <= 0 or class_depth >= len(path_parts):
            # If class_depth is 0 or greater than path depth, use the first directory
            return path_parts[0] if len(path_parts) > 1 else "unknown"
        
        # Return the class name at the specified depth
        return path_parts[class_depth - 1]
    except ValueError:
        return "unknown"


def is_clean_image(json_data: dict) -> bool:
    """
    Check if an image is considered "clean" based on overall_result.
    
    Args:
        json_data: Dictionary containing the JSON data
    
    Returns:
        True if the image is clean (overall_result is "PASS"), False otherwise
    """
    return json_data.get('overall_result', '').upper() == 'PASS'


def create_yolo_dataset_structure(
    images_data: List[Tuple[Path, Path, dict, str]],
    output_dir: Path,
    test_percent: float,
    val_percent: float,
    test_criteria: List[str],
    test_clean_percent: float,
    val_clean_percent: float
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Create YOLO dataset structure with train/val/test splits.
    
    Args:
        images_data: List of tuples (image_path, json_path, json_data, class_name)
        output_dir: Output directory
        test_percent: Percentage for test set split
        val_percent: Percentage for validation set split
        test_criteria: Criteria for test set selection
        test_clean_percent: Minimum percentage of clean images in test set
        val_clean_percent: Minimum percentage of clean images in validation set
    
    Returns:
        Tuple of (counts_dict, detailed_stats_dict)
    """
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate images into clean and test candidates
    clean_images = []
    test_candidates = []
    
    for image_path, json_path, json_data, class_name in images_data:
        if should_put_in_test(json_data, test_criteria):
            test_candidates.append((image_path, json_path, json_data, class_name))
        else:
            clean_images.append((image_path, json_path, json_data, class_name))
    
    print(f"Found {len(clean_images)} clean images and {len(test_candidates)} test candidates")
    
    # Shuffle clean images for random split
    random.shuffle(clean_images)
    
    # Calculate split sizes based on percentages
    total_clean = len(clean_images)
    test_size = max(1, int(total_clean * test_percent / 100))
    val_size = max(1, int(total_clean * val_percent / 100))
    
    # Ensure we don't exceed available images
    test_size = min(test_size, total_clean)
    val_size = min(val_size, total_clean - test_size)
    
    # Split clean images
    test_images = clean_images[:test_size]
    val_images = clean_images[test_size:test_size + val_size]
    train_images = clean_images[test_size + val_size:]
    
    # Process validation set - ensure minimum clean percentage
    val_clean_images = []
    val_other_images = []
    
    for img_data in val_images:
        if is_clean_image(img_data[2]):
            val_clean_images.append(img_data)
        else:
            val_other_images.append(img_data)
    
    # Calculate target clean images for validation
    target_clean_val = max(1, int(len(val_images) * val_clean_percent / 100))
    if len(val_clean_images) < target_clean_val:
        # Add more clean images from remaining data if needed
        remaining_clean = [img for img in train_images if is_clean_image(img[2])]
        needed_clean = target_clean_val - len(val_clean_images)
        val_clean_images.extend(remaining_clean[:needed_clean])
        # Remove the added images from train_images
        for img in remaining_clean[:needed_clean]:
            if img in train_images:
                train_images.remove(img)
    
    # Combine clean and other images for validation
    final_val_images = val_clean_images + val_other_images[:len(val_images) - len(val_clean_images)]
    
    # Process test set - ensure minimum clean percentage
    test_clean_images = []
    test_other_images = []
    
    for img_data in test_images:
        if is_clean_image(img_data[2]):
            test_clean_images.append(img_data)
        else:
            test_other_images.append(img_data)
    
    # Calculate target clean images for test
    target_clean_test = max(1, int(len(test_images) * test_clean_percent / 100))
    if len(test_clean_images) < target_clean_test:
        # Add more clean images from remaining data if needed
        remaining_clean = [img for img in train_images if is_clean_image(img[2])]
        needed_clean = target_clean_test - len(test_clean_images)
        test_clean_images.extend(remaining_clean[:needed_clean])
        # Remove the added images from train_images
        for img in remaining_clean[:needed_clean]:
            if img in train_images:
                train_images.remove(img)
    
    # Combine clean and other images for test
    final_test_images = test_clean_images + test_other_images[:len(test_images) - len(test_clean_images)]
    
    # Combine test candidates with final test images
    all_test_images = test_candidates + final_test_images
    
    # Process each split and collect detailed statistics
    splits = {
        'train': (train_images, train_dir),
        'val': (final_val_images, val_dir),
        'test': (all_test_images, test_dir)
    }
    
    counts = {}
    detailed_stats = {
        'train': {'total': 0, 'clean': 0, 'non_clean': 0, 'test_criteria': 0},
        'val': {'total': 0, 'clean': 0, 'non_clean': 0, 'test_criteria': 0},
        'test': {'total': 0, 'clean': 0, 'non_clean': 0, 'test_criteria': 0}
    }
    
    for split_name, (images, split_dir) in splits.items():
        print(f"\nProcessing {split_name} split ({len(images)} images)...")
        counts[split_name] = process_split(images, split_dir)
        
        # Analyze the split
        for image_path, json_path, json_data, class_name in images:
            detailed_stats[split_name]['total'] += 1
            if is_clean_image(json_data):
                detailed_stats[split_name]['clean'] += 1
            else:
                detailed_stats[split_name]['non_clean'] += 1
            if should_put_in_test(json_data, test_criteria):
                detailed_stats[split_name]['test_criteria'] += 1
    
    return counts, detailed_stats


def process_split(images_data: List[Tuple[Path, Path, dict, str]], split_dir: Path) -> int:
    """
    Process a single split (train/val/test) and copy images to the appropriate directory structure.
    
    Args:
        images_data: List of tuples (image_path, json_path, json_data, class_name)
        split_dir: Directory for this split
    
    Returns:
        Number of processed images
    """
    class_counters = defaultdict(int)
    processed_count = 0
    
    for image_path, json_path, json_data, class_name in images_data:
        try:
            # Create class directory
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate new filename
            counter = class_counters[class_name]
            new_filename = f"{class_name}_{counter:06d}.jpg"
            class_counters[class_name] += 1
            
            # Copy image to class directory with new name
            output_path = class_dir / new_filename
            shutil.copy2(image_path, output_path)
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} images...")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    return processed_count


def analyze_splits(images_data: List[Tuple[Path, Path, dict, str]], test_criteria: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Analyze the distribution of images across splits and their characteristics.
    
    Args:
        images_data: List of tuples (image_path, json_path, json_data, class_name)
        test_criteria: Criteria used for test set selection
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_images': len(images_data),
        'clean_images': 0,
        'non_clean_images': 0,
        'test_criteria_images': 0,
        'splits': {
            'train': {'total': 0, 'clean': 0, 'non_clean': 0, 'test_criteria': 0},
            'val': {'total': 0, 'clean': 0, 'non_clean': 0, 'test_criteria': 0},
            'test': {'total': 0, 'clean': 0, 'non_clean': 0, 'test_criteria': 0}
        }
    }
    
    for image_path, json_path, json_data, class_name in images_data:
        is_clean = is_clean_image(json_data)
        is_test_criteria = should_put_in_test(json_data, test_criteria)
        
        if is_clean:
            analysis['clean_images'] += 1
        else:
            analysis['non_clean_images'] += 1
            
        if is_test_criteria:
            analysis['test_criteria_images'] += 1
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to YOLO format with JSON-based filtering and train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_to_yolo.py --input-dir ../inpainted_images --json-dir ../vlm_screened_images --output-dir ./yolo_dataset --test-percent 10 --val-percent 10 --remove_criteria correct_image photorealistic --test-criteria white_background no_watermarks_blemishes --class-depth 2
    
    python convert_to_yolo.py --input-dir ./images --json-dir ./annotations --output-dir ./output --test-percent 15 --val-percent 15 --remove_criteria correct_image --class-depth 1
        """
    )
    
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing input images (supports recursive subdirectories)"
    )
    
    parser.add_argument(
        "--json-dir", 
        required=True,
        type=Path,
        help="Directory containing JSON annotation files (supports recursive subdirectories)"
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for YOLO dataset structure"
    )
    
    parser.add_argument(
        "--test-percent",
        type=float,
        default=10.0,
        help="Percentage of images for test set split (default: 10.0)"
    )
    
    parser.add_argument(
        "--val-percent",
        type=float,
        default=10.0,
        help="Percentage of images for validation set split (default: 10.0)"
    )
    
    parser.add_argument(
        "--test-clean-percent",
        type=float,
        default=0.0,
        help="Minimum percentage of clean images in test set (default: 0.0)"
    )
    
    parser.add_argument(
        "--val-clean-percent",
        type=float,
        default=90.0,
        help="Minimum percentage of clean images in validation set (default: 90.0)"
    )
    
    parser.add_argument(
        "--remove_criteria",
        nargs="+",
        default=[],
        help="JSON keys to check for exclusion. Images with 'no' values for these keys will be excluded"
    )
    
    parser.add_argument(
        "--test-criteria",
        nargs="+",
        default=[],
        help="JSON keys to check for test set. Images with 'no' values for these keys will be put in test set"
    )
    
    parser.add_argument(
        "--class-depth",
        type=int,
        default=1,
        help="Depth level for class name extraction (default: 1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate input directories
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return 1
    
    if not args.json_dir.exists():
        print(f"Error: JSON directory '{args.json_dir}' does not exist.")
        return 1
    
    # Validate percentages
    if args.test_percent + args.val_percent > 100:
        print(f"Error: test_percent ({args.test_percent}) + val_percent ({args.val_percent}) cannot exceed 100")
        return 1
    
    # Validate clean percentages
    if args.test_clean_percent > 100 or args.val_clean_percent > 100:
        print(f"Error: Clean percentages cannot exceed 100")
        return 1
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing images from: {args.input_dir}")
    print(f"Processing JSON files from: {args.json_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test split percentage: {args.test_percent}%")
    print(f"Validation split percentage: {args.val_percent}%")
    print(f"Test clean percentage: {args.test_clean_percent}%")
    print(f"Validation clean percentage: {args.val_clean_percent}%")
    print(f"Remove criteria: {args.remove_criteria}")
    print(f"Test criteria: {args.test_criteria}")
    print(f"Class depth: {args.class_depth}")
    print("-" * 50)
    
    # Find all image files recursively
    image_files = find_files_recursively(args.input_dir, '.jpg')
    print(f"Found {len(image_files)} image files")
    
    # Find all JSON files recursively
    json_files = find_files_recursively(args.json_dir, '.json')
    print(f"Found {len(json_files)} JSON files")
    
    # Create a mapping of JSON files by their base name (without extension)
    json_map = {}
    for json_file in json_files:
        base_name = json_file.stem
        json_map[base_name] = json_file
    
    print(f"Created mapping for {len(json_map)} JSON files")
    
    # Process each image file
    images_data = []
    excluded_count = 0
    missing_json_count = 0
    excluded_by_criteria = defaultdict(int)
    
    for image_file in image_files:
        # Get the base name of the image file (without extension)
        base_name = image_file.stem
        
        # Check if corresponding JSON file exists
        if base_name not in json_map:
            print(f"Warning: No JSON file found for image {image_file}")
            missing_json_count += 1
            continue
        
        json_file = json_map[base_name]
        
        try:
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Check if image should be excluded based on criteria
            if should_exclude_image(json_data, args.remove_criteria):
                # Track which criteria caused the exclusion
                for criterion in args.remove_criteria:
                    if criterion in json_data and json_data[criterion].lower() == 'no':
                        excluded_by_criteria[criterion] += 1
                        break
                print(f"Excluding {image_file} - failed criteria: {args.remove_criteria}")
                excluded_count += 1
                continue
            
            # Get class name based on class depth
            class_name = get_class_name_from_path(image_file, args.input_dir, args.class_depth)
            
            # Add to images_data
            images_data.append((image_file, json_file, json_data, class_name))
            
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {json_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    print(f"\nProcessed {len(images_data)} valid images")
    print(f"Excluded {excluded_count} images")
    print(f"Missing JSON: {missing_json_count} images")
    
    # Print exclusion breakdown
    if excluded_by_criteria:
        print(f"\nExclusion breakdown by criteria:")
        for criterion, count in excluded_by_criteria.items():
            print(f"  {criterion}: {count} images")
    
    if not images_data:
        print("No valid images found!")
        return 1
    
    # Analyze the dataset before splitting
    analysis = analyze_splits(images_data, args.test_criteria)
    print(f"\nDataset analysis:")
    print(f"  Total valid images: {analysis['total_images']}")
    print(f"  Clean images (PASS): {analysis['clean_images']}")
    print(f"  Non-clean images: {analysis['non_clean_images']}")
    print(f"  Test criteria images: {analysis['test_criteria_images']}")
    
    # Create YOLO dataset structure
    print("\nCreating YOLO dataset structure...")
    counts, detailed_stats = create_yolo_dataset_structure(
        images_data, 
        args.output_dir, 
        args.test_percent, 
        args.val_percent, 
        args.test_criteria,
        args.test_clean_percent,
        args.val_clean_percent
    )
    
    # Analyze final splits
    print("\n" + "="*50)
    print("FINAL DATASET STATISTICS")
    print("="*50)
    print(f"Dataset structure created in: {args.output_dir}")
    print(f"Train images: {counts['train']}")
    print(f"Validation images: {counts['val']}")
    print(f"Test images: {counts['test']}")
    print(f"Total images: {sum(counts.values())}")
    
    # Display detailed statistics for each split
    print(f"\nDetailed split statistics:")
    for split_name in ['train', 'val', 'test']:
        stats = detailed_stats[split_name]
        print(f"  {split_name.upper()} split:")
        print(f"    Total images: {stats['total']}")
        print(f"    Clean images: {stats['clean']} ({stats['clean']/stats['total']*100:.1f}%)")
        print(f"    Non-clean images: {stats['non_clean']} ({stats['non_clean']/stats['total']*100:.1f}%)")
        print(f"    Test criteria images: {stats['test_criteria']} ({stats['test_criteria']/stats['total']*100:.1f}%)")
    
    # Calculate and display non-clean image distribution
    if analysis['test_criteria_images'] > 0:
        print(f"\nNon-clean images (test criteria) distribution:")
        print(f"  Total non-clean images found: {analysis['test_criteria_images']}")
        print(f"  Distribution across splits:")
        for split_name in ['train', 'val', 'test']:
            stats = detailed_stats[split_name]
            print(f"    {split_name.upper()}: {stats['test_criteria']} images ({stats['test_criteria']/analysis['test_criteria_images']*100:.1f}%)")
    
    print(f"\nExclusion summary:")
    print(f"  Images excluded by criteria: {excluded_count}")
    print(f"  Images with missing JSON: {missing_json_count}")
    print(f"  Total excluded: {excluded_count + missing_json_count}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 