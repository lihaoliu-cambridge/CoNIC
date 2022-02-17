import sys
import logging
import os

import shutil
import argparse

import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.utils import io as IPyIO
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 300

# adding the project root folder
sys.path.append('../')
from tiatoolbox.models import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.utils.visualization import overlay_prediction_contours

from .misc.utils import cropping_center, recur_find_ext, rm_n_mkdir, rmdir

# Random seed for deterministic
SEED = 5
# The number of nuclei within the dataset/predictions.
# For CoNIC, we have 6 (+1 for background) types in total.
NUM_TYPES = 7
# The path to the directory containg images.npy etc.


def run_model(npy_file_path, output_dir_path, user_data_dir, fold=5):
    # The fold to use
    FOLD_IDX = int(fold)

    # The path to the pretrained weights
    print(f'{user_data_dir}/semantic_model/fold_{FOLD_IDX}/net_fold_{FOLD_IDX}.tar')
    PRETRAINED = f'{user_data_dir}/semantic_model/fold_{FOLD_IDX}/net_fold_{FOLD_IDX}.tar'
    # The path to contain output and intermediate processing results
    tmp_results_dir = os.path.join(output_dir_path, "tmp")

    # imgs = np.load(npy_file_path)

    rm_n_mkdir(f'{tmp_results_dir}/raw/')

    def convert_pytorch_checkpoint(net_state_dict):
        variable_name_list = list(net_state_dict.keys())
        is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
        if is_in_parallel_mode:
            print(
                (
                    " Detect checkpoint saved in data-parallel mode."
                    " Converting saved model to single GPU mode."
                ).rjust(80)
            )
            net_state_dict = {
                ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
            }
        return net_state_dict

    from .net_desc import HoVerNetConic

    if PRETRAINED.endswith(".pth"):
        pretrained = torch.load(PRETRAINED) # , map_location=torch.device('cpu'))
    else:
        net_state_dict = torch.load(PRETRAINED)["desc"] # , map_location=torch.device('cpu'))["desc"]
        pretrained = convert_pytorch_checkpoint(net_state_dict)
    model = HoVerNetConic(num_types=NUM_TYPES)
    model.load_state_dict(pretrained)

    # Tile prediction
    predictor = SemanticSegmentor(
        model=model,
        num_loader_workers=2,
        batch_size=480,
    )

    # Define the input/output configurations
    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {'units': 'baseline', 'resolution': 1.0},
        ],
        output_resolutions=[
            {'units': 'baseline', 'resolution': 1.0},
            {'units': 'baseline', 'resolution': 1.0},
            {'units': 'baseline', 'resolution': 1.0},
        ],
        save_resolution={'units': 'baseline', 'resolution': 1.0},
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
    )

    logger = logging.getLogger()
    logger.disabled = True

    infer_img_paths = recur_find_ext(f'{tmp_results_dir}/imgs/', ['.png'])
    rmdir(f'{tmp_results_dir}/raw/')

    # capture all the printing to avoid cluttering the console
    with IPyIO.capture_output() as captured:
        output_file = predictor.predict(
            infer_img_paths,
            masks=None,
            mode='tile',
            on_gpu=True,
            ioconfig=ioconfig,
            crash_on_exception=True,
            save_dir=f'{tmp_results_dir}/raw/'
        )





    def process_segmentation(np_map, hv_map, tp_map):
        # HoVerNet post-proc is coded at 0.25mpp so we resize
        np_map = cv2.resize(np_map, (0, 0), fx=1.0, fy=1.0)
        hv_map = cv2.resize(hv_map, (0, 0), fx=1.0, fy=1.0)
        tp_map = cv2.resize(
                        tp_map, (0, 0), fx=1.0, fy=1.0,
                        interpolation=cv2.INTER_NEAREST)

        inst_map = model._proc_np_hv(np_map[..., None], hv_map)
        inst_dict = model._get_instance_info(inst_map, tp_map)

        # Generating results match with the evaluation protocol
        type_map = np.zeros_like(inst_map)
        inst_type_colours = np.array([
            [v['type']] * 3 for v in inst_dict.values()
        ])
        type_map = overlay_prediction_contours(
            type_map, inst_dict,
            line_thickness=-1,
            inst_colours=inst_type_colours)

        pred_map = np.dstack([inst_map, type_map])
        # The result for evaluation is at 0.5mpp so we scale back
        pred_map = cv2.resize(
                        pred_map, (0, 0), fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_NEAREST)
        return pred_map

    def process_composition(pred_map):
        # Only consider the central 224x224 region,
        # as noted in the challenge description paper
        pred_map = cropping_center(pred_map, [224, 224])
        inst_map = pred_map[..., 0]
        type_map = pred_map[..., 1]
        # ignore 0-th index as it is 0 i.e background
        uid_list = np.unique(inst_map)[1:]

        if len(uid_list) < 1:
            type_freqs = np.zeros(NUM_TYPES)
            return type_freqs
        uid_types = [
            np.unique(type_map[inst_map == uid])
            for uid in uid_list
        ]
        type_freqs_ = np.unique(uid_types, return_counts=True)
        # ! not all types exist within the same spatial location
        # ! so we have to create a placeholder and put them there
        type_freqs = np.zeros(NUM_TYPES)
        type_freqs[type_freqs_[0]] = type_freqs_[1]
        return type_freqs

    output_file = f'{tmp_results_dir}/raw/file_map.dat'
    output_info = joblib.load(output_file)

    semantic_predictions = []
    composition_predictions = []
    for input_file, output_root in tqdm(output_info):
        img = cv2.imread(input_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np_map = np.load(f'{output_root}.raw.0.npy')
        hv_map = np.load(f'{output_root}.raw.1.npy')
        tp_map = np.load(f'{output_root}.raw.2.npy')

        pred_map = process_segmentation(np_map, hv_map, tp_map)
        type_freqs = process_composition(pred_map)
        semantic_predictions.append(pred_map)
        composition_predictions.append(type_freqs)
    semantic_predictions = np.array(semantic_predictions)
    composition_predictions = np.array(composition_predictions)





    # Saving the results for segmentation
    print(f"save to {output_dir_path}/tmp/results/semantic_pred_fold_{FOLD_IDX}.npy")
    np.save(f'{output_dir_path}/tmp/results/semantic_pred_fold_{FOLD_IDX}.npy', semantic_predictions)

    rm_n_mkdir(f'{tmp_results_dir}/raw/')
    # rm_n_mkdir(f'{tmp_results_dir}/imgs/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--npy_file_path', type=str)
    parser.add_argument('--output_dir_path', type=str)
    parser.add_argument('--user_data_dir', type=str)
    parser.add_argument('--fold', type=str)
    args = parser.parse_args()

    run_model(args.npy_file_path, args.output_dir_path, args.user_data_dir, args.fold)
