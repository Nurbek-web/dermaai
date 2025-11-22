"""
Grouped Split Script for HAM10000 Dataset
==========================================
Creates lesion-level stratified splits to prevent data leakage.
No images from the same lesion appear in both train and validation sets.

Usage:
    python grouped_split.py

Outputs:
    - data/splits/train_val_split.csv (lesion_id -> partition mapping)
    - data/splits/split_summary.txt (statistics and verification)
    - Regenerated data/processed/ folders with grouped split
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import shutil
import os

# Configuration
SEED = 42
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1

DATA_DIR = Path('data')
METADATA_PATH = DATA_DIR / 'HAM10000_metadata.tab'
IMAGES_PART_1_PATH = DATA_DIR / 'HAM10000_images_part_1'
IMAGES_PART_2_PATH = DATA_DIR / 'HAM10000_images_part_2'

SPLITS_DIR = DATA_DIR / 'splits'
OUTPUT_DIR = DATA_DIR / 'processed_grouped'
TRAIN_DIR = OUTPUT_DIR / 'train'
VAL_DIR = OUTPUT_DIR / 'val'

# Lesion type mapping
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',  # Keep trailing space for consistency
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def load_and_prepare_metadata():
    """Load metadata and prepare for grouped splitting."""
    print("Loading metadata...")
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(df)} image records")
    
    # Add readable lesion type
    df['cell_type'] = df['dx'].map(lesion_type_dict)
    
    # Add image paths
    df['path'] = df['image_id'].apply(
        lambda x: str(IMAGES_PART_1_PATH / f'{x}.jpg') 
        if (IMAGES_PART_1_PATH / f'{x}.jpg').exists() 
        else str(IMAGES_PART_2_PATH / f'{x}.jpg')
    )
    
    # Verify all images exist
    missing_images = df[~df['path'].apply(lambda x: Path(x).exists())]
    if len(missing_images) > 0:
        print(f"WARNING: {len(missing_images)} images not found!")
        df = df[df['path'].apply(lambda x: Path(x).exists())]
        print(f"Proceeding with {len(df)} images")
    
    return df

def create_grouped_split(df):
    """Create lesion-level stratified split."""
    print("\nCreating grouped split...")
    
    # Get unique lesions with their dominant class
    lesion_summary = df.groupby('lesion_id').agg({
        'dx': lambda x: x.mode()[0],  # Most common class for this lesion
        'cell_type': lambda x: x.mode()[0],
        'image_id': 'count'  # Number of images per lesion
    }).reset_index()
    lesion_summary.columns = ['lesion_id', 'dx', 'cell_type', 'n_images']
    
    print(f"Total unique lesions: {len(lesion_summary)}")
    print(f"Images per lesion - Mean: {lesion_summary['n_images'].mean():.1f}, "
          f"Median: {lesion_summary['n_images'].median():.1f}, "
          f"Max: {lesion_summary['n_images'].max()}")
    
    # Use GroupShuffleSplit for stratified lesion-level split
    gss = GroupShuffleSplit(
        n_splits=1, 
        train_size=TRAIN_RATIO, 
        random_state=SEED
    )
    
    # Create stratification labels (lesion class)
    y_lesions = lesion_summary['dx'].values
    groups = lesion_summary['lesion_id'].values
    
    # Get train/val lesion indices
    train_idx, val_idx = next(gss.split(lesion_summary, y_lesions, groups))
    
    # Get train/val lesion IDs
    train_lesions = set(lesion_summary.iloc[train_idx]['lesion_id'])
    val_lesions = set(lesion_summary.iloc[val_idx]['lesion_id'])
    
    # Verify no overlap
    assert len(train_lesions & val_lesions) == 0, "ERROR: Lesion overlap detected!"
    
    # Assign partition to each image
    df['partition'] = df['lesion_id'].map(
        lambda x: 'train' if x in train_lesions else 'val'
    )
    
    return df, train_lesions, val_lesions

def verify_split(df):
    """Verify split quality and print statistics."""
    print("\n" + "="*50)
    print("SPLIT VERIFICATION")
    print("="*50)
    
    train_df = df[df['partition'] == 'train']
    val_df = df[df['partition'] == 'val']
    
    print(f"Total images: {len(df)}")
    print(f"Train images: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val images: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    
    # Verify no lesion overlap
    train_lesions = set(train_df['lesion_id'].unique())
    val_lesions = set(val_df['lesion_id'].unique())
    overlap = train_lesions & val_lesions
    
    print(f"\nLesion-level verification:")
    print(f"Train lesions: {len(train_lesions)}")
    print(f"Val lesions: {len(val_lesions)}")
    print(f"Overlap: {len(overlap)} ✓" if len(overlap) == 0 else f"Overlap: {len(overlap)} ❌")
    
    # Class distribution
    print(f"\nClass distribution:")
    print("Train:")
    train_dist = train_df['cell_type'].value_counts(normalize=True)
    for cls, pct in train_dist.items():
        print(f"  {cls}: {pct:.3f}")
    
    print("Val:")
    val_dist = val_df['cell_type'].value_counts(normalize=True)
    for cls, pct in val_dist.items():
        print(f"  {cls}: {pct:.3f}")
    
    # Calculate distribution difference
    print(f"\nDistribution differences (|train% - val%|):")
    for cls in train_dist.index:
        diff = abs(train_dist[cls] - val_dist.get(cls, 0))
        print(f"  {cls}: {diff:.3f}")
    
    return len(overlap) == 0

def organize_images(df):
    """Organize images into train/val directories by class."""
    print("\nOrganizing images into directories...")
    
    # Remove existing output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    # Create directory structure
    for partition in ['train', 'val']:
        for cell_type in df['cell_type'].unique():
            (OUTPUT_DIR / partition / cell_type).mkdir(parents=True, exist_ok=True)
    
    # Copy images
    for _, row in df.iterrows():
        src_path = Path(row['path'])
        dst_path = OUTPUT_DIR / row['partition'] / row['cell_type'] / src_path.name
        shutil.copy2(src_path, dst_path)
    
    print(f"Images organized in: {OUTPUT_DIR}")
    
    # Print final counts
    print(f"\nFinal directory structure:")
    for partition in ['train', 'val']:
        print(f"  {partition}/")
        partition_df = df[df['partition'] == partition]
        for cell_type in sorted(df['cell_type'].unique()):
            count = len(partition_df[partition_df['cell_type'] == cell_type])
            print(f"    {cell_type}: {count} images")

def save_split_info(df, train_lesions, val_lesions):
    """Save split information for reproducibility."""
    SPLITS_DIR.mkdir(exist_ok=True)
    
    # Save lesion -> partition mapping
    lesion_partition_df = df[['lesion_id', 'partition']].drop_duplicates()
    lesion_partition_df.to_csv(SPLITS_DIR / 'train_val_split.csv', index=False)
    
    # Save detailed summary
    with open(SPLITS_DIR / 'split_summary.txt', 'w') as f:
        f.write("HAM10000 Grouped Split Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Split date: {pd.Timestamp.now()}\n")
        f.write(f"Random seed: {SEED}\n")
        f.write(f"Train ratio: {TRAIN_RATIO}\n")
        f.write(f"Val ratio: {VAL_RATIO}\n\n")
        
        f.write(f"Total images: {len(df)}\n")
        f.write(f"Total lesions: {len(train_lesions) + len(val_lesions)}\n")
        f.write(f"Train lesions: {len(train_lesions)}\n")
        f.write(f"Val lesions: {len(val_lesions)}\n\n")
        
        train_df = df[df['partition'] == 'train']
        val_df = df[df['partition'] == 'val']
        
        f.write("Class distribution:\n")
        for cls in sorted(df['cell_type'].unique()):
            train_count = len(train_df[train_df['cell_type'] == cls])
            val_count = len(val_df[val_df['cell_type'] == cls])
            total_count = train_count + val_count
            f.write(f"  {cls}:\n")
            f.write(f"    Train: {train_count} ({train_count/len(train_df)*100:.1f}%)\n")
            f.write(f"    Val: {val_count} ({val_count/len(val_df)*100:.1f}%)\n")
            f.write(f"    Total: {total_count}\n")
    
    print(f"\nSplit info saved to: {SPLITS_DIR}")

def main():
    """Main execution function."""
    print("HAM10000 Grouped Split Generator")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    # Load and prepare data
    df = load_and_prepare_metadata()
    
    # Create grouped split
    df, train_lesions, val_lesions = create_grouped_split(df)
    
    # Verify split quality
    split_valid = verify_split(df)
    
    if not split_valid:
        print("❌ Split validation failed! Exiting.")
        return
    
    # Organize images
    organize_images(df)
    
    # Save split information
    save_split_info(df, train_lesions, val_lesions)
    
    print("\n✅ Grouped split completed successfully!")
    print(f"Use the images in: {OUTPUT_DIR}")
    print(f"Split info available in: {SPLITS_DIR}")

if __name__ == "__main__":
    main()
