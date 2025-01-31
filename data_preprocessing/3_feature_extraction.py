import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


def extract_image_features(image_folder, output_file, target_dim=1024):
    """
    Load images from a given folder, extract image features using a pre-trained ResNet50 model,
    reduce the feature dimension to 1024, and save the result as an .npy file.

    Parameters:
    image_folder (str): Path to the folder containing images.
    output_file (str): Path to save the extracted features, default is 'image_features.npy'.
    target_dim (int): Final feature dimension, default is 1024.
    """

    # 1. Define image preprocessing steps (removing CenterCrop)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Load the pre-trained ResNet50 model and remove the final classification layer
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()  # Set to evaluation mode

    # 3. Add a linear dimensionality reduction layer to reduce 2048D features to 1024D
    linear_layer = torch.nn.Linear(2048, target_dim)
    linear_layer = linear_layer.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Check if GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 5. Define a dictionary to store image names and their corresponding features
    features_dict = {}

    # 6. Iterate through all images in the folder
    for img_name in tqdm(os.listdir(image_folder)):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            # 7. Load image
            img_path = os.path.join(image_folder, img_name)
            try:
                # 7. Load image
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)
                img_tensor = img_tensor.to(device)

                # 8. Use ResNet50 to extract 2048D features
                with torch.no_grad():
                    feature_2048 = model(img_tensor).squeeze()
                    feature_1024 = linear_layer(feature_2048).cpu().numpy()

            except Exception as e:
                print(f"Skipping file {img_name}, error: {e}")

            features_dict[img_name[:-4]] = feature_1024

    np.save(output_file, features_dict)

    print(f"Feature extraction completed and saved as '{output_file}'")


dataset = ['Ours']
for data in dataset:

    img_roots = [
                 f'./Our_HD_data/{data}/16_information/cropped_img_224',
                 f'./Our_HD_data/{data}/16_information/cropped_img_512']

    save_roots = [
                  f'./Our_HD_data/{data}/extracted_feature/224',
                  f'./Our_HD_data/{data}/extracted_feature/512']

    for i in range(4):
        sample_folders = img_roots[i]
        save_sample_folders = save_roots[i]

        img_folders = sample_folders
        save_folders = save_sample_folders
        if not os.path.exists(save_folders):
            os.mkdir(save_folders)

        for save_img in os.listdir(img_folders):
            save_dirs = os.path.join(save_folders, save_img)
            img_dirs = os.path.join(img_folders, save_img)

            save_paths = os.path.join(save_folders, f'{save_img}.npy')
            extract_image_features(img_dirs, save_paths)