from PIL import Image
import os
import pandas as pd
import io
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def generate_patches(image_path, output_dir, patch_size, output_csv_dir, thred):
    # Open a PNG image
    img = Image.open(image_path)
    width, height = img.size  # Get the width and height of the image

    # Calculate the patch step size to ensure the last patch fits exactly to the edge
    step_x = width // patch_size[0]
    step_y = height // patch_size[1]

    patches = []
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    save_dir = os.path.join(output_dir, image_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f" {save_dir}")

    # Traverse the image and crop patches based on the step size
    for i in tqdm(range(step_x)):
        for j in tqdm(range(step_y)):
            # Calculate the top-left coordinates of the patch
            x = i * patch_size[0]
            y = j * patch_size[1]

            # Ensure the patch does not exceed the bottom-right edge of the image
            if x + patch_size[0] > width:
                x = width - patch_size[0]
            if y + patch_size[1] > height:
                y = height - patch_size[1]

            # Crop the patch
            patch = img.crop((x, y, x + patch_size[0], y + patch_size[1]))

            # Save the cropped image to memory and compute its size
            buffer = io.BytesIO()
            patch.save(buffer, format="PNG")
            file_size = buffer.tell()  # File size in bytes

            # Set a size threshold (e.g., 10KB)
            size_threshold = thred  # 10 KB

            # Decide whether to save
            if file_size >= size_threshold:
                patch_filename = f"{image_name}_size_{patch_size[0]}_{x}_{y}.png"
                patch.save(os.path.join(save_dir, patch_filename))

                # Record patch information, including coordinates i and j
                patches.append({
                    'patch_filename': patch_filename,
                    'image_name': image_name,
                    'x': x,
                    'y': y,
                    'i': i,
                    'j': j,
                    'width': patch_size[0],
                    'height': patch_size[1]
                })

    # Generate the CSV file path for the corresponding image
    csv_file = os.path.join(output_csv_dir, f"{image_name}_patches_{patch_size[0]}.csv")

    # Save all patch information to a separate CSV file
    df = pd.DataFrame(patches)
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved: {csv_file}")


def batch_process_images(image_dir, output_dir_, csv_dir_):
    # Get the paths of all PNG images
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        generate_patches(image_path, output_dir_, patch_size=(224, 224), output_csv_dir=csv_dir_, thred=45 * 1024)


if __name__ == "__main__":
    # Input folder path containing PNG images
    images = "./Our_HD_data/Ours/image"

    # Output folder path for patches
    output_dir_224_root = "./Our_HD_data/Ours/cropped_img_224"
    output_dir_512_root = "./Our_HD_data/Ours/cropped_img_512"
    # Output folder path for CSV files (storing patch information for each image)
    csv_dir_224_root = "./Our_HD_data/Ours/csv_infor_224"
    csv_dir_512_root = "./Our_HD_data/Ours/csv_infor_512"

    for sample in os.listdir(images):
        img = os.path.join(images, sample)
        output_dir_224 = output_dir_224_root
        output_dir_512 = output_dir_512_root

        csv_dir_224 = csv_dir_224_root
        csv_dir_512 = csv_dir_512_root

        # Create output folders (if they do not exist)
        os.makedirs(output_dir_224, exist_ok=True)
        os.makedirs(output_dir_512, exist_ok=True)

        # Create CSV folders (if they do not exist)
        os.makedirs(csv_dir_224, exist_ok=True)
        os.makedirs(csv_dir_512, exist_ok=True)

        # Batch process PNG images, generating 1024x1024 and 512x512 patches and saving them in separate CSV files
        batch_process_images(img, output_dir_224, csv_dir_224)
