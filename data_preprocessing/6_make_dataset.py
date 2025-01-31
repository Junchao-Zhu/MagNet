import numpy as np
import pandas as pd
import os


def find_region(x, y, region_size=512):
    # Calculate the region index for point (x, y)
    region_x = x // region_size
    region_y = y // region_size
    return region_x, region_y


root_dir = './patches/112'
for sample_dir in os.listdir(root_dir):
    img_file_dir = f'./patches/112/{sample_dir}'
    gene_expression_dir = f'./gene_expression/112/{sample_dir}'
    feature_224_path = f'./extracted_feature/224/{sample_dir}'
    feature_512_path = f'./extracted_feature/512/{sample_dir}'
    gene_expression_dir_224 = f'./gene_expression/224/{sample_dir}'
    gene_expression_dir_512 = f'./gene_expression/512/{sample_dir}'
    output_path = f'./npy_information/{sample_dir}'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for label_file in os.listdir(gene_expression_dir):
        img_name = label_file[:-4]
        tmp_label_file_112 = pd.read_csv(os.path.join(gene_expression_dir, label_file))
        position_file_112 = pd.read_csv(os.path.join(img_file_dir, label_file))
        tmp_feature_224 = np.load(os.path.join(feature_224_path, f'{img_name}.npy'), allow_pickle=True).item()
        tmp_feature_512 = np.load(os.path.join(feature_512_path, f'{img_name}.npy'), allow_pickle=True).item()
        tmp_label_224 = pd.read_csv(os.path.join(gene_expression_dir_224, f'{img_name}_patches_224.csv'))
        tmp_label_512 = pd.read_csv(os.path.join(gene_expression_dir_512, f'{img_name}_patches_512.csv'))

        information_list = []
        for spot in tmp_label_file_112.columns[1:]:
            information_dict = {}
            img_path = f'./112/{img_name}/{spot}.png'

            label = np.array(tmp_label_file_112[spot])

            x_series = position_file_112.loc[position_file_112.iloc[:, 0] == spot]['i']
            y_series = position_file_112.loc[position_file_112.iloc[:, 0] == spot]['j']

            # Check if Series is empty
            if not x_series.empty and not y_series.empty:
                try:
                    position = [int(x_series.iloc[0]), int(y_series.iloc[0])]
                except (ValueError, TypeError) as e:
                    print(f"Skipping due to conversion error: {e}")
                    continue
            else:
                print(f"Skipping because x or y Series is empty for spot {spot}")
                continue

            true_position = [int(position_file_112.loc[position_file_112.iloc[:, 0] == spot]['X']),
                             int(position_file_112.loc[position_file_112.iloc[:, 0] == spot]['Y'])]
            pair_224 = find_region(true_position[0], true_position[1], region_size=224)
            pair_512 = find_region(true_position[0], true_position[1], region_size=512)

            pair_224_img_name = f'{img_name}_size_224_{int(pair_224[0]*224)}_{int(pair_224[1]*224)}.png'
            pair_512_img_name = f'{img_name}_size_512_{int(pair_512[0] * 512)}_{int(pair_512[1] * 512)}.png'

            if pair_224_img_name in tmp_label_224.columns and pair_512_img_name in tmp_label_512.columns:
                label_224 = tmp_label_224[pair_224_img_name]
                feature_224 = tmp_feature_224[pair_224_img_name[:-4]]
                label_512 = tmp_label_512[pair_512_img_name]
                feature_512 = tmp_feature_512[pair_512_img_name[:-4]]

                information_dict['img_path'] = img_path
                information_dict['label'] = label
                information_dict['position'] = position
                information_dict['feature_224'] = feature_224
                information_dict['feature_512'] = feature_512
                information_dict['label_224'] = np.array(label_224)
                information_dict['label_512'] = np.array(label_512)

                information_list.append(information_dict)

        save_path = os.path.join(output_path, f'{img_name}.npy')
        print(save_path)
        print(len(information_list))
        np.save(save_path, information_list)
