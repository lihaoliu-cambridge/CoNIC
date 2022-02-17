"""compute_stats.py. Calculates the statistical measurements for the CoNIC Challenge.

This code supports binary panoptic quality for binary segmentation, multiclass panoptic quality for 
simultaneous segmentation and classification and multiclass coefficient of determination (R2) for
multiclass regression. Binary panoptic quality is calculated per image and the results are averaged.
For multiclass panoptic quality, stats are calculated over the entire dataset for each class before taking
the average over the classes.

Usage:
    compute_stats.py [--mode=<str>] [--pred=<path>] [--true=<path>]
    compute_stats.py (-h | --help)
    compute_stats.py --version

Options:
    -h --help                   Show this string.
    --version                   Show version.
    --mode=<str>                Choose either `regression` or `seg_class`.
    --pred=<path>               Path to the results directory.
    --true=<path>               Path to the ground truth directory.

"""

from docopt import docopt
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
from ensemble_boxes_nms import nms
import argparse


def remove_zeros(pred_array_list):
    nr_patches = pred_array_list.shape[0]
    for idx in range(nr_patches):
        pred_instance_and_type = pred_array_list[idx]
        pred_instance_map = pred_instance_and_type[..., 0]
        pred_type_map = pred_instance_and_type[..., 1]
        for instance_id in np.unique(pred_instance_map):
            if instance_id == 0:
                continue
            pred_2_instance_part = (pred_instance_map == instance_id)
            pred_2_category_ids_in_instance = np.unique(pred_type_map[pred_2_instance_part])
            assert len(pred_2_category_ids_in_instance) == 1
            pred_2_category_id = int(pred_2_category_ids_in_instance[0])
            if pred_2_category_id > 6 or pred_2_category_id == 0:
                pred_instance_map[pred_2_instance_part] = 0
    return pred_array_list


def mask2box(mask):
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    y1 = int(np.min(rows))  # y
    x1 = int(np.min(clos))  # x
    y2 = int(np.max(rows))
    x2 = int(np.max(clos)) 
    return (x1, y1, x2, y2) 


def models_cv_masks_boxes_nms(models_cv_masks_boxes, threshold=0.5):
    boxes = np.concatenate(models_cv_masks_boxes)
    boxes_nms, pick, pick_all = non_max_suppression_fast(boxes, threshold)
    return boxes_nms, pick, pick_all


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    pick_all = []

    # grab the coordinates of the bounding boxes
    # print(boxes.shape, boxes[:,0].shape)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # print("sorted idx:", idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        # print('i',i,  x1[i], x1[idxs[:last]], np.maximum(x1[i], x1[idxs[:last]]))
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        # print(idxs[np.concatenate(([last], np.where(overlap > overlapThresh)[0]))])
        pick_all.append(idxs[np.concatenate(([last], np.where(overlap > overlapThresh)[0]))])

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the integer data type
    # return boxes[pick].astype("int"), pick, pick_all
    return None, pick, pick_all


def get_masks_from_nms_pick(pick_all, test_masks_cv_array, test_labels_cv_array, ensembled_array_idx):
    for instance_id, picked_cells in enumerate(pick_all, start=1):
        # print(picked_cells.shape)
        masks_from_different_models = test_masks_cv_array[picked_cells]
        labels_from_different_models = test_labels_cv_array[picked_cells]
        
        # print(masks_from_different_models.shape, masks_from_different_models.dtype)
        mask_array = masks_from_different_models[0]
        # for i in range(1, masks_from_different_models.shape[0]):
        #     mask_array = np.logical_or(mask_array, masks_from_different_models[i])
        # print(mask.shape, mask.dtype)

        # print(labels_from_different_models)
        label = np.argmax(np.bincount(labels_from_different_models))
        # print(label)

        ensembled_array_idx[..., 0][mask_array] = instance_id
        ensembled_array_idx[..., 1][mask_array] = int(label)
    
    return ensembled_array_idx


def get_regression_results(pred_array, OUT_DIR):
    # Recalculate Counts
    middle_seg = pred_array[:, 16:240, 26:240, :]

    all_counts = []
    # print(middle_seg.shape[0])
    for i in range(middle_seg.shape[0]):
        cell_counts = [0,0,0,0,0,0]

        instance_and_type = middle_seg[i]
        instance_map = instance_and_type[..., 0]
        type_map = instance_and_type[..., 1]
            
        instance_ids = np.unique(instance_map)
        for instance_id in instance_ids:
            if instance_id == 0:
                continue

            # category_id
            instance_part = (instance_map == instance_id)
            category_ids_in_instance = np.unique(type_map[instance_part])
            if len(category_ids_in_instance) != 1:
                type_map[instance_part] = np.argmax(np.bincount(type_map[instance_part].astype(np.int64)))
                category_ids_in_instance = np.unique(type_map[instance_part])
            assert len(category_ids_in_instance) == 1
            category_id = int(category_ids_in_instance[0])
            if category_id > 6 or category_id == 0:
                continue
                # raise Exception("Only 6 types")

            cell_counts[category_id-1] += 1
        all_counts.append(cell_counts)
        
    all_counts_2_np = np.asarray(all_counts)
    df = pd.DataFrame(data=all_counts_2_np, columns=["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"])
    df.to_csv(f"{OUT_DIR}/pred.csv", index=False)
    print(f"save to {OUT_DIR}/pred.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--pred_fold_0', type=str)
    parser.add_argument('--pred_fold_1', type=str)
    parser.add_argument('--pred_fold_2', type=str)
    parser.add_argument('--pred_fold_3', type=str)
    parser.add_argument('--pred_fold_4', type=str)
    parser.add_argument('--out_dir_path', type=str)
    args = parser.parse_args()

    pred_fold_0_path = args.pred_fold_0
    pred_fold_1_path = args.pred_fold_1
    pred_fold_2_path = args.pred_fold_2
    pred_fold_3_path = args.pred_fold_3
    pred_fold_4_path = args.pred_fold_4
    OUT_DIR = args.out_dir_path
    
    # pred_fold_0_array = remove_zeros(np.load(pred_fold_0_path))
    # pred_fold_1_array = remove_zeros(np.load(pred_fold_1_path))
    # pred_fold_2_array = remove_zeros(np.load(pred_fold_2_path))
    # pred_fold_3_array = remove_zeros(np.load(pred_fold_3_path))
    # pred_fold_4_array = remove_zeros(np.load(pred_fold_4_path))
    
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
            scores_list = []
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

    np.save(f'{OUT_DIR}/pred.npy', ensembled_array)
    print(f"save to {OUT_DIR}/pred.npy")

    get_regression_results(ensembled_array, OUT_DIR)