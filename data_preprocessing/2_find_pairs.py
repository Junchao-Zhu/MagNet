import pandas as pd
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
import os


def find_patches_containing_points(points_csv, patches_csv, patch_size, num_jobs=128):
    """
    Assign points to their corresponding patches.

    Parameters:
    - points_csv: Path to the Parquet file containing point information.
    - patches_csv: Path to the CSV file containing patch information.
    - patch_size: Size of the patch (assuming each patch is a square).
    - num_jobs: Number of parallel processing jobs.

    Returns:
    - patch_points_dict: A dictionary where the keys are patch names and the values are lists of point names.
    """
    # Read the Parquet file containing point information
    points_df = pd.read_parquet(points_csv)
    column_names = ['barcode', '_', 'i', 'j', 'pixel_x', 'pixel_y']
    points_df.columns = column_names

    # Read the CSV file containing patch information
    patches_df = pd.read_csv(patches_csv)

    # Compute the maximum boundaries for each patch
    patches_df['x_max'] = patches_df['x'] + patch_size
    patches_df['y_max'] = patches_df['y'] + patch_size

    # Convert patch data to a NumPy array for vectorized operations
    patches_array = patches_df[['patch_filename', 'x', 'y', 'x_max', 'y_max']].to_numpy()

    # Convert point data to a NumPy array
    points_array = points_df[['barcode', 'pixel_x', 'pixel_y']].to_numpy()

    # Define a function to process each chunk
    def process_chunk(chunk):
        local_dict = defaultdict(list)
        # Extract point coordinates and names
        barcodes = chunk[:, 0]
        pixel_x = chunk[:, 1].astype(np.float64)
        pixel_y = chunk[:, 2].astype(np.float64)

        # Find the patch that contains each point
        for i in tqdm(range(len(chunk))):
            x = pixel_x[i]
            y = pixel_y[i]

            # Vectorized condition check to find matching patches
            mask = (patches_array[:, 1] <= x) & (x < patches_array[:, 3]) & \
                   (patches_array[:, 2] <= y) & (y < patches_array[:, 4])

            matched_patches = patches_array[mask]

            if matched_patches.size > 0:
                # Assume each point belongs to only one patch (the first matching patch)
                patch_name = matched_patches[0][0]
                point_name = barcodes[i]
                local_dict[patch_name].append(point_name)

        return local_dict

    # Split point data into multiple chunks
    chunks = np.array_split(points_array, num_jobs)

    # Process each chunk in parallel and display progress with tqdm
    results = Parallel(n_jobs=num_jobs)(
        delayed(process_chunk)(chunk) for chunk in tqdm(chunks, desc="Processing chunks")
    )

    # Merge results from all chunks
    patch_points_dict = defaultdict(list)
    for local_dict in results:
        for patch, points in local_dict.items():
            patch_points_dict[patch].extend(points)

    return patch_points_dict


def update_patches_csv_with_points(patches_csv, patch_points_dict):
    # Read the CSV file for 512x512 patches
    patches_df = pd.read_csv(patches_csv)

    # Add a new column for each patch to record the included point names
    patches_df['points'] = patches_df['patch_filename'].apply(
        lambda patch_name: ', '.join(patch_points_dict[patch_name]))

    # Save the updated CSV file
    patches_df.to_csv(patches_csv, index=False)
    print(f"Updated CSV: {patches_csv}")


def remove_patches_without_points(patches_csv, patch_points_dict, patch_dir):
    # Read the CSV file for 512x512 patches
    patches_df = pd.read_csv(patches_csv)

    # Find patches that do not contain any points
    patches_without_points = [patch for patch, points in patch_points_dict.items() if not points]
    print(patches_without_points)

    # Delete patch images in the folder that do not contain points
    for patch_filename in patches_without_points:
        patch_path = os.path.join(patch_dir, patch_filename)

        if os.path.exists(patch_path):
            os.remove(patch_path)
            print(f"Deleted: {patch_path}")

    # Remove records of patches that do not contain points from the CSV file
    patches_df_filtered = patches_df[~patches_df['patch_filename'].isin(patches_without_points)]

    # Save the updated CSV file
    patches_df_filtered.to_csv(patches_csv, index=False)
    print(f"Updated CSV: {patches_csv}")


def batch_process_images_with_points(image_dir, points_dir, patches_dir, patch_size, patch_output_dir):
    # Get paths of all PNG images
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in tqdm(image_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Get image filename without extension
        print(f"Image: {image_name}")

        # Corresponding point CSV file path
        points_csv = os.path.join(points_dir, f"{image_name}.parquet")

        # Corresponding patch CSV file path
        patches_csv = os.path.join(patches_dir, f"{image_name}_patches_{patch_size}.csv")

        # Corresponding patch image folder
        patch_dir = os.path.join(patch_output_dir, image_name)

        # Find patches that contain points and their corresponding points
        patch_points_dict = find_patches_containing_points(points_csv, patches_csv, patch_size)

        # Update CSV file by adding a column for contained point names
        update_patches_csv_with_points(patches_csv, patch_points_dict)

        # Remove patches without points and their records from the CSV file
        remove_patches_without_points(patches_csv, patch_points_dict, patch_dir)


if __name__ == "__main__":
    idxs = ['08', '16']
    patch_sizes = [224, 512]

    for idx in idxs:
        for patch_size in patch_sizes:
            images = './Our_HD_data/Ours/image'
            points_dir = f'./Our_HD_data/Ours/location_{idx}'
            patches_dir = f'./Our_HD_data/Ours/{idx}_information/csv_infor_{patch_size}'
            patch_output_dir = f'./Our_HD_data/Ours/{idx}_information/cropped_img_{patch_size}'

            # Batch process PNG images, handle corresponding points for each image, and update patch information
            batch_process_images_with_points(images, points_dir, patches_dir, patch_size, patch_output_dir)