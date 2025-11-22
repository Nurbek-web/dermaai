import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path('data')
METADATA_PATH = DATA_DIR / 'HAM10000_metadata.tab'
IMAGES_PART_1_PATH = DATA_DIR / 'HAM10000_images_part_1'
IMAGES_PART_2_PATH = DATA_DIR / 'HAM10000_images_part_2'
OUTPUT_DIR = DATA_DIR / 'processed'
TRAIN_DIR = OUTPUT_DIR / 'train'
VAL_DIR = OUTPUT_DIR / 'val'

# Lesion type dictionary for mapping short codes to full names
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def main():
    """
    Main function to orchestrate the data preparation process.
    """
    print("Starting data preparation...")

    # --- 1. Load and Prepare Metadata ---
    df = pd.read_csv(METADATA_PATH)
    print(f"Loaded metadata with {len(df)} records.")

    # Create a path column for easier access
    df['path'] = df['image_id'].apply(lambda x: str(IMAGES_PART_1_PATH / f'{x}.jpg') if os.path.exists(IMAGES_PART_1_PATH / f'{x}.jpg') else str(IMAGES_PART_2_PATH / f'{x}.jpg'))
    
    # Create a readable label column
    df['cell_type'] = df['dx'].map(lesion_type_dict.get)
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

    print("Metadata prepared with image paths and labels.")

    # --- 2. Create Train/Validation Split ---
    # We use a 90/10 split as described in the paper.
    # Stratify ensures that the class distribution is the same in train and val sets.
    y = df['cell_type']
    _, df_val = train_test_split(df, test_size=0.10, random_state=42, stratify=y)
    
    # The rest of the data is for training. However, to easily move files,
    # we'll identify the training set by excluding validation image_ids.
    val_ids = set(df_val['image_id'])
    df_train = df[~df['image_id'].isin(val_ids)]
    
    print(f"Data split into {len(df_train)} training and {len(df_val)} validation samples.")
    print("\nTraining set distribution:")
    print(df_train['cell_type'].value_counts(normalize=True))
    print("\nValidation set distribution:")
    print(df_val['cell_type'].value_counts(normalize=True))


    # --- 3. Organize Images into Directories ---
    print("\nOrganizing images into train and validation directories...")

    # Remove existing directories to ensure a clean start
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Removed existing directory: {OUTPUT_DIR}")

    # Create the directory structure
    for class_name in df['cell_type'].unique():
        os.makedirs(TRAIN_DIR / class_name, exist_ok=True)
        os.makedirs(VAL_DIR / class_name, exist_ok=True)

    # Copy training images
    for _, row in df_train.iterrows():
        src_path = Path(row['path'])
        dst_path = TRAIN_DIR / row['cell_type'] / src_path.name
        shutil.copyfile(src_path, dst_path)

    # Copy validation images
    for _, row in df_val.iterrows():
        src_path = Path(row['path'])
        dst_path = VAL_DIR / row['cell_type'] / src_path.name
        shutil.copyfile(src_path, dst_path)
    
    print("Successfully organized images.")
    print(f"Training images are in: {TRAIN_DIR}")
    print(f"Validation images are in: {VAL_DIR}")


if __name__ == '__main__':
    # Unzip the image folders first if they are not already unzipped
    # This is a manual step for the user.
    # Assumes 'HAM10000_images_part_1.zip' and 'HAM10000_images_part_2.zip' are in 'data/'
    # and have been unzipped into 'HAM10000_images_part_1' and 'HAM10000_images_part_2'
    
    # Check if image directories exist
    if not (IMAGES_PART_1_PATH.exists() and IMAGES_PART_2_PATH.exists()):
        print("Error: Image directories not found.")
        print(f"Please make sure you have unzipped 'HAM10000_images_part_1.zip' and 'HAM10000_images_part_2.zip'")
        print(f"into the '{DATA_DIR}' directory.")
    else:
        main()

