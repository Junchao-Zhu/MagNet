import os
import numpy as np
from sklearn.model_selection import KFold


def four_fold_cross_validation(data_folder):
    """
    Implements four-fold cross-validation by dividing the .npy files in the folder into training and testing sets.

    :param data_folder: Path to the data folder containing .npy files
    :return: A list of four-fold training and testing set paths, where each fold contains a training set and a testing set
    """
    # Get all .npy file paths
    npy_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.npy')]

    # Shuffle the file list to ensure a more randomized split
    np.random.shuffle(npy_files)

    # Initialize KFold
    kf = KFold(n_splits=4, shuffle=False)  # shuffle=False ensures a fixed split method

    # Store training and testing paths for each fold
    folds = []
    for train_index, test_index in kf.split(npy_files):
        train_files = [npy_files[i] for i in train_index]
        test_files = [npy_files[i] for i in test_index]
        folds.append((train_files, test_files))

    return folds

