import os
import sys
import cv2
import itk
import json
from tqdm import tqdm
import shutil
import argparse
import numpy as np


def rm_n_mkdir(dir_path):
    """Remove and then make a new directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def generate_jpg_from_npy(npy_file_path, output_dir_path):
    tmp_result_dir_path = os.path.join(output_dir_path, "tmp")
    image_dir_path = os.path.join(tmp_result_dir_path, "imgs")
    json_dir_path = os.path.join(tmp_result_dir_path, "json")
    results_dir_path = os.path.join(tmp_result_dir_path, "results")
    rm_n_mkdir(tmp_result_dir_path)
    rm_n_mkdir(image_dir_path)
    rm_n_mkdir(json_dir_path)
    rm_n_mkdir(results_dir_path)

    # imgs = np.load(npy_file_path)
    imgs = np.array(itk.imread(npy_file_path))

    # for line in enumerate(tqdm(f)):
    for idx in tqdm(list(range(imgs.shape[0]))):
        # print(idx)
        img = imgs[idx]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_png = cv2.resize(img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f'{tmp_result_dir_path}/imgs/{idx:06d}.png', img_png)

        img_jpg = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(f'{tmp_result_dir_path}/imgs/{idx:06d}.jpg', img_jpg)
    
    generate_coco_format_dataset("test", imgs, image_dir_path=image_dir_path, json_dir_path=json_dir_path)


def generate_coco_format_dataset(phase, imgs, image_dir_path, json_dir_path):
    result = {
        "info": {"description": "CoNIC test dataset."},
        "categories": [
            {'id': 1, 'name': 'neutrophil'},
            {'id': 2, 'name': 'epithelial'},
            {'id': 3, 'name': 'lymphocyte'},
            {'id': 4, 'name': 'plasma'},
            {'id': 5, 'name': 'eosinophil'},
            {'id': 6, 'name': 'connective'}
        ]
    }

    images_info = []
    for idx in list(range(imgs.shape[0])):
        # Images
        image_name = f"{idx:06d}.jpg"

        images_info.append(
            {
                "file_name": image_name,
                "height": 512,
                "width": 512,
                "id": idx
            }
        )

    result["images"] = images_info

    json_file_path = os.path.join(json_dir_path, 'instances_{}2017.json'.format(phase))
    with open(json_file_path, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--npy_file_path', type=str)
    parser.add_argument('--output_dir_path', type=str)
    args = parser.parse_args()

    generate_jpg_from_npy(args.npy_file_path, args.output_dir_path)

