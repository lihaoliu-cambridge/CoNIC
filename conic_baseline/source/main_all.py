

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
    
    generate_jpg_from_npy(IMG_PATH, output_dir)

    # #  ------------------ Step 2 ------------------ 
    from .run_instance_model import default_argument_parser, register_coco_instances, _get_coco_instances_meta, main, launch
    
    OUTPUT_DIR_PATH = output_dir
    register_coco_instances("test_dataset", _get_coco_instances_meta(), f"{OUTPUT_DIR_PATH}/tmp/json/instances_test2017.json", f"{OUTPUT_DIR_PATH}/tmp/imgs")

    for fold_idx_instance in range(5):
        params_list = ["--json_file_path", f"{OUTPUT_DIR_PATH}/tmp/json/instances_test2017.json", "--image_dir_path", f"{OUTPUT_DIR_PATH}/tmp/imgs", "--output_dir_path", f"{OUTPUT_DIR_PATH}", "--num-gpus", "1", "--fold", "{fold_idx_instance}", "--config-file", f"{user_data_dir}/instance_model/fold_{fold_idx_instance}/config_fold_{fold_idx_instance}.yaml", "--eval-only", "MODEL.WEIGHTS", f"{user_data_dir}/instance_model/fold_{fold_idx_instance}/model_fold_{fold_idx_instance}.pth"]
        args_instance = default_argument_parser().parse_args(params_list)
        print("Command Line Args:", args_instance)

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

    for fold_idx_semantic in range(5):
        run_model(IMG_PATH, OUTPUT_DIR_PATH, user_data_dir, fold_idx_semantic)

    #  ------------------  Step 4 ------------------ 
    from .overwrite_ensemble import overwrite_ensemble_two_maps_current_best_simplified, get_regression_results
    
    for fold_idx_ensemble in range(5):
        instance_pred_path = f"{OUTPUT_DIR_PATH}/tmp/results/instance_pred_fold_{fold_idx_ensemble}.npy"
        semantic_pred_path = f"{OUTPUT_DIR_PATH}/tmp/results/semantic_pred_fold_{fold_idx_ensemble}.npy"
        
        instance_pred_format = instance_pred_path.split(".")[-1]
        semantic_pred_format = semantic_pred_path.split(".")[-1]
        if instance_pred_format != "npy" or semantic_pred_format != "npy":
            raise ValueError("pred and true must be in npy format.")

        instance_pred_array = np.load(instance_pred_path)
        senmatic_pred_array = np.load(semantic_pred_path)

        ensembled_pred_array = overwrite_ensemble_two_maps_current_best_simplified(senmatic_pred_array, instance_pred_array)
        np.save(f'{OUTPUT_DIR_PATH}/tmp/results/pred_fold_{fold_idx_ensemble}.npy', ensembled_pred_array)

    #  ------------------  Step 5 ------------------ 
    from .fold_ensemble import mask2box, models_cv_masks_boxes_nms, get_masks_from_nms_pick

    pred_fold_0_path = f"{OUTPUT_DIR_PATH}/tmp/results/pred_fold_0.npy"
    pred_fold_1_path = f"{OUTPUT_DIR_PATH}/tmp/results/pred_fold_1.npy"
    pred_fold_2_path = f"{OUTPUT_DIR_PATH}/tmp/results/pred_fold_2.npy"
    pred_fold_3_path = f"{OUTPUT_DIR_PATH}/tmp/results/pred_fold_3.npy"
    pred_fold_4_path = f"{OUTPUT_DIR_PATH}/tmp/results/pred_fold_4.npy"
    OUT_DIR = OUTPUT_DIR_PATH
    
    pred_fold_0_array = np.load(pred_fold_0_path)
    pred_fold_1_array = np.load(pred_fold_1_path)
    pred_fold_2_array = np.load(pred_fold_2_path)
    pred_fold_3_array = np.load(pred_fold_3_path)
    pred_fold_4_array = np.load(pred_fold_4_path)

    ensembled_array = np.zeros(pred_fold_0_array.shape)

    # ensemble on each image‘s results
    for idx in range(pred_fold_0_array.shape[0]):
        bbox_fold_list = []
        labels_fold_list = []
        mask_fold_list = []

        # each model's prediction
        for pred_array in [pred_fold_0_array, pred_fold_1_array, pred_fold_2_array, pred_fold_3_array, pred_fold_4_array]:
            pred_instance_and_type = pred_array[idx]
            pred_instance_map = pred_instance_and_type[..., 0]
            pred_type_map = pred_instance_and_type[..., 1]

            bbox_list = []
            labels_list = []
            mask_list = []

            pred_instance_ids = np.unique(pred_instance_map)
            # find_bounding_boxes_on_masksssssssss
            for instance_id in pred_instance_ids:
                if instance_id == 0:
                    continue
                
                boxes = []
                # category_id
                instance_part = (pred_instance_map == instance_id)
                category_ids_in_instance = np.unique(pred_type_map[instance_part])
                assert len(category_ids_in_instance) == 1
                category_id = int(category_ids_in_instance[0])
                if category_id > 6 or category_id == 0:
                    raise Exception("Only 6 types")

                # bbox
                x1, y1, x2, y2 = mask2box(instance_part)
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                
                boxes = (x1, y1, w, h)
                boxes_np = np.array(boxes)
         
                bbox_list.append(boxes_np)
                mask_list.append(instance_part)
                labels_list.append(category_id)

            # print("This model detects {} cells.".format(len(bbox_list)))
            if len(bbox_list) == 0 and len(mask_list) == 0 and len(labels_list) == 0:
                # print("No cells in this fold.")
                continue

            bbox_fold_list.append(bbox_list)
            mask_fold_list.append(mask_list)
            labels_fold_list.append(labels_list)
        
        if len(bbox_fold_list) == 0 and len(labels_fold_list) == 0 and len(labels_fold_list) == 0:
            # print("No cells in all fold of this image.")
            continue

        _, pick, pick_all = models_cv_masks_boxes_nms(bbox_fold_list, threshold=0.5)

        ensembled_array_idx = get_masks_from_nms_pick(pick_all, np.concatenate(mask_fold_list), np.concatenate(labels_fold_list), ensembled_array[idx])
        ensembled_array[idx, ...] = ensembled_array_idx

    semantic_predictions = ensembled_array

    # For regression results
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
