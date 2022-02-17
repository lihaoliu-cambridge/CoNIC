

import itk
import logging
import os
import warnings

warnings.filterwarnings("ignore")

import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

from .net_desc import HoVerNetConic
from .utils import (
    cropping_center, overlay_prediction_contours,
    recur_find_ext, rm_n_mkdir, rmdir, print_dir,
    save_as_json)


def process_composition(pred_map, num_types):
    # Only consider the central 224x224 region,
    # as noted in the challenge description paper
    pred_map = cropping_center(pred_map.astype(np.int), [224, 224])
    inst_map = pred_map[..., 0]
    type_map = pred_map[..., 1]
    # ignore 0-th index as it is 0 i.e background
    uid_list = np.unique(inst_map)[1:]

    if len(uid_list) < 1:
        type_freqs = np.zeros(num_types)
        return type_freqs
    uid_types = [
        np.unique(type_map[inst_map == uid])
        for uid in uid_list
    ]
    type_freqs_ = np.unique(uid_types, return_counts=True)
    # ! not all types exist within the same spatial location
    # ! so we have to create a placeholder and put them there
    type_freqs = np.zeros(num_types)
    type_freqs[type_freqs_[0]] = type_freqs_[1]
    return type_freqs


def run(
        input_dir: str,
        output_dir: str,
        user_data_dir: str,
    ) -> None:
    """Entry function for automatic evaluation.

    This is the function which will be called by the organizer
    docker template to trigger evaluation run. All the data
    to be evaluated will be provided in "input_dir" while
    all the results that will be measured must be saved
    under "output_dir". Participant auxiliary data is provided
    under  "user_data_dir".

    input_dir (str): Path to the directory which contains input data.
    output_dir (str): Path to the directory which will contain output data.
    user_data_dir (str): Path to the directory which contains user data. This
        data include model weights, normalization matrix etc. .

    """
    # ===== Header script for user checking
    print(f"INPUT_DIR: {input_dir}")
    # recursively print out all subdirs and their contents
    print_dir(input_dir)
    print("USER_DATA_DIR: ", os.listdir(user_data_dir))
    # recursively print out all subdirs and their contents
    print_dir(user_data_dir)
    print(f"OUTPUT_DIR: {output_dir}")

    print(f"CUDA: {torch.cuda.is_available()}")
    for device in range(torch.cuda.device_count()):
        print(f"---Device {device}: {torch.cuda.get_device_name(0)}")

    paths = recur_find_ext(f"{input_dir}", [".mha"])
    assert len(paths) == 1, "There should only be one image package."
    IMG_PATH = paths[0]

    # convert from .mha to .npy
    images = np.array(itk.imread(IMG_PATH))
    # np.save("images.npy", images)

    # ===== Whatever you need

    start_time = time.time()

    # #  ------------------ Step 1 ------------------ 
    from .generate_jpg_file import generate_jpg_from_npy
    # generate_jpg_from_npy(IMG_PATH, output_dir)

    # #  ------------------ Step 2 ------------------ 
    from .run_instance_model import default_argument_parser, register_coco_instances, _get_coco_instances_meta, main, launch
    OUTPUT_DIR_PATH = output_dir
    params_list = ["--json_file_path", f"{OUTPUT_DIR_PATH}/tmp/json/instances_test2017.json", "--image_dir_path", f"{OUTPUT_DIR_PATH}/tmp/imgs", "--output_dir_path", f"{OUTPUT_DIR_PATH}", "--num-gpus", "1", "--fold", "5", "--config-file", f"{user_data_dir}/instance_model/fold_5/config_fold_5.yaml", "--eval-only", "MODEL.WEIGHTS", f"{user_data_dir}/instance_model/fold_5/model_fold_5.pth"]

    args_instance = default_argument_parser().parse_args(params_list)
    print("Command Line Args:", args_instance)

    register_coco_instances("test_dataset", _get_coco_instances_meta(), args_instance.json_file_path, args_instance.image_dir_path)

    launch(
        main,
        args_instance.num_gpus,
        num_machines=args_instance.num_machines,
        machine_rank=args_instance.machine_rank,
        dist_url=args_instance.dist_url,
        args=(args_instance,),
    )

    # # ------------------ Step 3 ------------------ 
    from .run_semantic_model import run_model
    run_model(IMG_PATH, OUTPUT_DIR_PATH, user_data_dir, 5)

    #  ------------------  Step 4 ------------------ 
    from .overwrite_ensemble import overwrite_ensemble_two_maps_current_best_simplified, get_regression_results
    instance_pred_path = f"{OUTPUT_DIR_PATH}/tmp/results/instance_pred_fold_5.npy"
    semantic_pred_path = f"{OUTPUT_DIR_PATH}/tmp/results/semantic_pred_fold_5.npy"
    OUT_DIR = OUTPUT_DIR_PATH
    FOLD_IDX = 5
    
    instance_pred_format = instance_pred_path.split(".")[-1]
    semantic_pred_format = semantic_pred_path.split(".")[-1]
    if instance_pred_format != "npy" or semantic_pred_format != "npy":
        raise ValueError("pred and true must be in npy format.")

    instance_pred_array = np.load(instance_pred_path)
    senmatic_pred_array = np.load(semantic_pred_path)

    ensembled_pred_array = overwrite_ensemble_two_maps_current_best_simplified(senmatic_pred_array, instance_pred_array)
    semantic_predictions = ensembled_pred_array

    # # For regression
    # if int(FOLD_IDX) == 5:
    #     np.save(f'{OUT_DIR}/pred.npy', ensembled_pred_array)
    #     print(f"save to {OUT_DIR}/pred.npy")
    # else:
    #     np.save(f'{OUT_DIR}/tmp/results/pred_fold_{FOLD_IDX}.npy', ensembled_pred_array)
    #     print(f"save to {OUT_DIR}/tmp/results/pred_fold_{FOLD_IDX}.npy")

    # get_regression_results(ensembled_pred_array, OUT_DIR=OUT_DIR, FOLD_IDX=FOLD_IDX)

    composition_predictions = []
    # for input_file, output_root in tqdm(output_info):
    num_images = semantic_predictions.shape[0]
    NUM_TYPES = 7
    for idx in range(num_images):
        pred_map = semantic_predictions[idx, ...]
        type_freqs = process_composition(
            pred_map, NUM_TYPES)

        composition_predictions.append(type_freqs)
    composition_predictions = np.array(composition_predictions)

    # ! >>>>>>>>>>>> Saving to approriate format for evaluation docker

    # Saving the results for segmentation in .mha
    itk.imwrite(
        itk.image_from_array(semantic_predictions),
        f"{OUT_DIR}/pred_seg.mha"
    )

    # version v0.0.8
    # Saving the results for composition prediction
    TYPE_NAMES = [
        "neutrophil",
        "epithelial-cell",
        "lymphocyte",
        "plasma-cell",
        "eosinophil",
        "connective-tissue-cell"
    ]
    for type_idx, type_name in enumerate(TYPE_NAMES):
        cell_counts = composition_predictions[:, (type_idx+1)]
        cell_counts = cell_counts.astype(np.int32).tolist()
        save_as_json(
            cell_counts,
            f'{OUT_DIR}/{type_name}-count.json'
        )

    TYPE_NAMES = [
        "neutrophil", "epithelial", "lymphocyte",
        "plasma", "eosinophil", "connective"
    ]
    df = pd.DataFrame(
        composition_predictions[:, 1:].astype(np.int32),
    )
    df.columns = TYPE_NAMES
    df.to_csv(f'{OUT_DIR}/pred_count.csv', index=False)

    end_time = time.time()
    print("Run time: ", end_time - start_time)

    # ! <<<<<<<<<<<<
