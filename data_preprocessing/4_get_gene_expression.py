import os

import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# Define global variables for access in child processes
element_to_index_global = {}
mtx_global = None
last_col_global = ''
delimiter_global = ','

def init_worker(element_to_index, mtx, last_col, delimiter):
    """
    Initialize the child process and set global variables.
    """
    global element_to_index_global
    global mtx_global
    global last_col_global
    global delimiter_global
    element_to_index_global = element_to_index
    mtx_global = mtx
    last_col_global = last_col
    delimiter_global = delimiter

def process_row(row):
    """
    Function to process a single row, used by child processes.
    """
    # Retrieve the last column elements, assuming they are stored as a delimited string
    elements_str = row[last_col_global]
    # Split into individual elements and remove potential whitespace
    elements = [elem.strip() for elem in str(elements_str).split(delimiter_global)]

    # Retrieve the indices of corresponding elements in the Parquet file, skipping non-existent elements
    indices = []
    for elem in elements:
        if elem in element_to_index_global:
            idx = element_to_index_global[elem]
            if idx < mtx_global.shape[1]:
                indices.append(idx)
            else:
                # Skip the element if the index is out of range
                pass
        else:
            # Skip the element if it is not present in the Parquet file
            pass

    if not indices:
        # If no indices were found, return a zero vector
        sum_vector = np.zeros(mtx_global.shape[0])
    else:
        # Select the corresponding columns and compute the sum
        selected_columns = mtx_global[:, indices]
        # Convert to a dense matrix and sum along the axis
        sum_vector = selected_columns.sum(axis=1).A1  # A1 converts it to a 1D NumPy array

    return sum_vector


def process_files(csv_path, parquet_path, mtx_path, output_npy_path, delimiter=','):
    """
    Process CSV, Parquet, and MTX files, and generate a NumPy array saved as an .npy file.

    Parameters:
    - csv_path: Path to the CSV file
    - parquet_path: Path to the Parquet file
    - mtx_path: Path to the MTX file
    - output_npy_path: Path to save the output .npy file
    - delimiter: Delimiter for elements in the last column of the CSV file (default is comma)
    """
    # 1. Load the CSV file
    csv_df = pd.read_csv(csv_path)
    # Retrieve the column name of the last column
    last_col = csv_df.columns[-1]

    # 2. Load the Parquet file
    parquet_df = pd.read_parquet(parquet_path)
    # Get the column name of the first column in the Parquet file
    parquet_id_col = parquet_df.columns[0]
    element_to_index = {element: idx for idx, element in enumerate(parquet_df[parquet_id_col])}

    mtx = mmread(mtx_path)
    # Ensure the MTX file is in CSR sparse matrix format
    if not isinstance(mtx, csr_matrix):
        mtx = mtx.tocsr()

    # 4. Check if the number of elements in the Parquet file matches the number of columns in the MTX file
    num_parquet_elements = len(parquet_df[parquet_id_col])
    num_mtx_columns = mtx.shape[1]

    if num_parquet_elements != num_mtx_columns:
        print("Error: shape not match")
        # Handle mismatch as needed, such as continuing processing or adjusting mapping

    result_list = [None] * len(csv_df)

    # Get the number of CPU cores
    num_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker,
                             initargs=(element_to_index, mtx, last_col, delimiter)) as executor:
        # Submit all tasks and store row indices
        futures = {executor.submit(process_row, row): idx for idx, row in csv_df.iterrows()}

        # Monitor progress using tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
            idx = futures[future]
            try:
                result = future.result()
                result_list[idx] = result
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                result_list[idx] = np.zeros(mtx.shape[0])

    try:
        result_array = np.vstack(result_list)
        print(f"Result npy shape: {result_array.shape}")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 7. Save as .npy file
    print(result_array.shape)
    print(f"Saving result to {output_npy_path}...")
    try:
        np.save(output_npy_path, result_array)
        print("Finished！")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    dataset = ['Ours', 'CRC']
    idxs = ['08', '16']
    levels = [224, 512]

    for data in dataset:
        for idx in idxs:
            for level in levels:
                for file in os.listdir(f'./Our_HD_data/{data}/gene_expression/original/16'):
                    tmp_name = file[:-4]
                    csv_file_path = f"./Our_HD_data/{data}/{idx}_information/csv_infor_{level}/{tmp_name}_patches_{level}.csv"
                    parquet_file_path = f"./Our_HD_data/{data}/location_16/in_tissue/{tmp_name}.parquet"
                    mtx_file_path = f"./Our_HD_data/{data}/gene_expression/original/16/{tmp_name}.mtx"
                    output_npy_file_path = f"./Our_HD_data/{data}/16_information/gene_expression_{level}/original/{tmp_name}.npy"

                    process_files(csv_file_path, parquet_file_path, mtx_file_path, output_npy_file_path)