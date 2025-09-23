#!/usr/bin/env python3
import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm

def calculate_md5(filepath, chunk_size=8192):
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def find_duplicate_images(directory):
    """Find duplicate images in a directory based on MD5 hash."""
    # Dictionary to store hash -> list of file paths
    hash_to_files = defaultdict(list)

    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(directory).glob(f"*{ext}"))
        image_files.extend(Path(directory).glob(f"*{ext.upper()}"))

    print(f"Found {len(image_files)} image files to analyze...")

    # Calculate hash for each file
    for filepath in tqdm(image_files, desc="Calculating hashes"):
        try:
            file_hash = calculate_md5(filepath)
            hash_to_files[file_hash].append(str(filepath))
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    # Find duplicates (hashes with more than one file)
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}

    return duplicates, len(image_files)

def analyze_duplicates(duplicates):
    """Analyze and report duplicate statistics."""
    total_duplicate_groups = len(duplicates)
    total_duplicate_files = sum(len(files) for files in duplicates.values())
    total_unique_duplicates = sum(len(files) - 1 for files in duplicates.values())

    # Get file size information for duplicates
    duplicate_info = []
    total_wasted_space = 0

    for hash_val, files in duplicates.items():
        file_size = os.path.getsize(files[0])
        wasted_space = file_size * (len(files) - 1)
        total_wasted_space += wasted_space

        duplicate_info.append({
            'hash': hash_val,
            'count': len(files),
            'file_size': file_size,
            'wasted_space': wasted_space,
            'files': files
        })

    # Sort by number of duplicates
    duplicate_info.sort(key=lambda x: x['count'], reverse=True)

    return {
        'total_groups': total_duplicate_groups,
        'total_duplicate_files': total_duplicate_files,
        'total_redundant_copies': total_unique_duplicates,
        'total_wasted_space_mb': total_wasted_space / (1024 * 1024),
        'duplicate_groups': duplicate_info
    }

def main():
    directory = "/Volumes/T7/SpheroSeg_upload/SpheroMix/train/images"

    print(f"Analyzing duplicates in: {directory}")
    print("-" * 80)

    # Find duplicates
    duplicates, total_files = find_duplicate_images(directory)

    if not duplicates:
        print("\n✅ No duplicate images found!")
        print(f"Total unique images: {total_files}")
        return

    # Analyze results
    stats = analyze_duplicates(duplicates)

    # Print summary
    print("\n" + "=" * 80)
    print("DUPLICATE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total images analyzed: {total_files:,}")
    print(f"Unique images: {total_files - stats['total_redundant_copies']:,}")
    print(f"Duplicate images (redundant copies): {stats['total_redundant_copies']:,}")
    print(f"Duplicate groups: {stats['total_groups']:,}")
    print(f"Wasted disk space: {stats['total_wasted_space_mb']:.2f} MB")
    print(f"Duplicate percentage: {(stats['total_redundant_copies'] / total_files * 100):.2f}%")

    # Show top duplicate groups
    print("\n" + "-" * 80)
    print("TOP 10 MOST DUPLICATED IMAGES")
    print("-" * 80)

    for i, group in enumerate(stats['duplicate_groups'][:10], 1):
        print(f"\n{i}. Group with {group['count']} copies (hash: {group['hash'][:16]}...)")
        print(f"   File size: {group['file_size']:,} bytes")
        print(f"   Wasted space: {group['wasted_space']:,} bytes")
        print(f"   Sample files:")
        for j, filepath in enumerate(group['files'][:3], 1):
            filename = os.path.basename(filepath)
            print(f"      {j}. {filename}")
        if len(group['files']) > 3:
            print(f"      ... and {len(group['files']) - 3} more")

    # Save detailed results to JSON
    output_file = "duplicate_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n📄 Detailed results saved to: {output_file}")

    # Create a list of files to potentially remove (keeping one of each duplicate)
    files_to_remove = []
    for group in stats['duplicate_groups']:
        # Keep the first file, mark others for removal
        files_to_remove.extend(group['files'][1:])

    removal_list_file = "duplicate_files_to_remove.txt"
    with open(removal_list_file, 'w') as f:
        for filepath in sorted(files_to_remove):
            f.write(f"{filepath}\n")
    print(f"📄 List of duplicate files (for potential removal) saved to: {removal_list_file}")
    print(f"   Total files that could be removed: {len(files_to_remove):,}")

if __name__ == "__main__":
    main()