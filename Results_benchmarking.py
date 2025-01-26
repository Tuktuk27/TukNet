import sys
import numpy as np
from itertools import product
from typing import List, Tuple
import cv2
import torch
from random import randrange
from utils.SFSORT import SFSORT
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import utils

print('Function to import from utils to be checked ')

def saving_video_benchmark(nn_name, path_saving, ref_label_path, detections_history, tracker_history, fps_history, img_size,
                           frame_num, tracker_type, video_path, iou_tresh = 0.5): # 'Luxonis/Kalman-Filer and Hungarian Algo'
    NN_WIDTH, NN_HEIGHT = img_size
    os.makedirs(path_saving, exist_ok=True)

    name_saving = os.path.basename(nn_name).split('.')[0]

    labels = []
    prev_box = []
    with open(ref_label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            bboxes = line.split(' | ')
            bboxes = [list(map(float, bbox.split())) for bbox in bboxes]
            if len(bboxes) > 2:
                print('too many drones in this label')
            labels.append(bboxes)

    print(f'{len(fps_history) = }')
    print(f'{len(detections_history) = }')
    print(f'{len(labels) = }')
    print(f'{frame_num = }')

    accuracy_history = []
    iou_history = []


    (iou_history, TPs, FPs, FNs, big_object_TPs, big_object_FPs, big_object_FNs, medium_object_TPs, medium_object_FPs,
     medium_object_FNs, small_object_TPs, small_object_FPs, small_object_FNs) = accu_iou_comput(detections_history, labels, iou_tresh)

    print(f'{sum(big_object_TPs), sum(medium_object_TPs), sum(small_object_TPs) = }')

    (tracker_iou_history, tracker_TPs, tracker_FPs, tracker_FNs, tracker_big_object_TPs, tracker_big_object_FPs,
     tracker_big_object_FNs, tracker_medium_object_TPs, tracker_medium_object_FPs, tracker_medium_object_FNs,
     tracker_small_object_TPs, tracker_small_object_FPs, tracker_small_object_FNs) = accu_iou_comput(tracker_history, labels, iou_tresh)

    object_num = sum(len(label) for label in labels) # 2 * frame_num should be equal to sum(len(label) for label in labels)

    precision = sum(TPs)/(sum(TPs)+sum(FPs)) if sum(TPs) !=0 else -1
    recall = sum(TPs)/(sum(TPs)+sum(FNs)) if sum(TPs) !=0 else -1

    big_object_precision = sum(big_object_TPs) / (sum(big_object_TPs) + sum(big_object_FPs)) if sum(big_object_TPs) !=0 else -1
    big_object_recall = sum(big_object_TPs) / (sum(big_object_TPs) + sum(big_object_FNs)) if sum(big_object_TPs) !=0 else -1

    medium_object_precision = sum(medium_object_TPs) / (sum(medium_object_TPs) + sum(medium_object_FPs)) if sum(medium_object_TPs) !=0 else -1
    medium_object_recall = sum(medium_object_TPs) / (sum(medium_object_TPs) + sum(medium_object_FNs)) if sum(medium_object_TPs) !=0 else -1

    small_object_precision = sum(small_object_TPs) / (sum(small_object_TPs) + sum(small_object_FPs)) if sum(small_object_TPs) !=0 else -1
    small_object_recall = sum(small_object_TPs) / (sum(small_object_TPs) + sum(small_object_FNs)) if sum(small_object_TPs) !=0 else -1

    tracker_precision = sum(tracker_TPs) / (sum(tracker_TPs) + sum(tracker_FPs)) if sum(tracker_TPs) !=0 else -1
    tracker_recall = sum(tracker_TPs) / (sum(tracker_TPs) + sum(tracker_FNs)) if sum(tracker_TPs) !=0 else -1

    tracker_big_object_precision = sum(tracker_big_object_TPs) / (sum(tracker_big_object_TPs) + sum(tracker_big_object_FPs)) if sum(tracker_big_object_TPs) !=0 else -1
    tracker_big_object_recall = sum(tracker_big_object_TPs) / (sum(tracker_big_object_TPs) + sum(tracker_big_object_FNs)) if sum(tracker_big_object_TPs) !=0 else -1

    tracker_medium_object_precision = sum(tracker_medium_object_TPs) / (sum(tracker_medium_object_TPs) + sum(tracker_medium_object_FPs)) if sum(tracker_medium_object_TPs) !=0 else -1
    tracker_medium_object_recall = sum(tracker_medium_object_TPs) / (sum(tracker_medium_object_TPs) + sum(tracker_medium_object_FNs)) if sum(tracker_medium_object_TPs) !=0 else -1

    tracker_small_object_precision = sum(tracker_small_object_TPs) / (sum(tracker_small_object_TPs) + sum(tracker_small_object_FPs)) if sum(tracker_small_object_TPs) !=0 else -1
    tracker_small_object_recall = sum(tracker_small_object_TPs) / (sum(tracker_small_object_TPs) + sum(tracker_small_object_FNs)) if sum(tracker_small_object_TPs) !=0 else -1

    F1_score = 2*recall*precision/(recall+precision)

    with open(f'{path_saving}/{video_path}_results_{NN_WIDTH}.txt', 'a') as f:
        mean_fps = f' || Fps mean: {(sum(fps_history) / len(fps_history)) :.3f} || '
        mean_precision = f'Precision: {precision :.3f} || ' # 2 * frame_num should be equal to sum(len(label) for label in labels)
        big_object_mean_precision = f'Big object Precision: {big_object_precision :.3f} || '  # 2 * frame_num should be equal to sum(len(label) for label in labels)
        medium_object_mean_precision = f'Medium object Precision: {medium_object_precision :.3f} || '  # 2 * frame_num should be equal to sum(len(label) for label in labels)
        small_object_mean_precision = f'Small object Precision: {small_object_precision :.3f} || '  # 2 * frame_num should be equal to sum(len(label) for label in labels)
        mean_recall = f'Recall: {recall :.3f} || '  # 2 * frame_num should be equal to sum(len(label) for label in labels)
        big_object_mean_recall = f'Big object Recall: {big_object_recall :.3f} || '  # 2 * frame_num should be equal to sum(len(label) for label in labels)
        medium_object_mean_recall = f'Medium object Recall: {medium_object_recall :.3f} || '  # 2 * frame_num should be equal to sum(len(label) for label in labels)
        small_object_mean_recall = f'Small object Recall: {small_object_recall :.3f} || '  # 2 * frame_num should be equal to sum(len(label) for label in labels)
        mean_iou = f'IOU mean: {sum(iou_history) / len(iou_history) :.3f} || '
        tracker_mean_precision = f'Precision with tracker: {tracker_precision:.3f} || '
        big_object_tracker_mean_precision = f'Big object Precision with tracker: {tracker_big_object_precision:.3f} || '
        medium_object_tracker_mean_precision = f'Medium object Precision with tracker: {tracker_medium_object_precision:.3f} || '
        small_object_tracker_mean_precision = f'Small object Precision with tracker: {tracker_small_object_precision:.3f} || '
        tracker_mean_recall = f'Recall with tracker: {tracker_recall:.3f} || '
        big_object_tracker_mean_recall = f'Big object Recall with tracker: {tracker_big_object_recall:.3f} || '
        medium_object_tracker_mean_recall = f'Medium object Recall with tracker: {tracker_medium_object_recall:.3f} || '
        small_object_tracker_mean_recall = f'Small object Recall with tracker: {tracker_small_object_recall:.3f} || '
        tracker_mean_iou = f'IOU mean with tracker: {sum(tracker_iou_history) / len(iou_history) :.3f} || '
        Image_size = f'Image size: {NN_WIDTH, NN_HEIGHT = } || '
        tracker_type_str = f'Tracker type: {tracker_type} || '
        F1_score_str = f'F1_score: {F1_score}'
        results = ' '.join(['model name: ', name_saving, mean_fps, mean_precision, big_object_mean_precision,medium_object_mean_precision,small_object_mean_precision, mean_recall, big_object_mean_recall, medium_object_mean_recall, small_object_mean_recall, tracker_mean_precision, big_object_tracker_mean_precision,medium_object_tracker_mean_precision,small_object_tracker_mean_precision, tracker_mean_recall, big_object_tracker_mean_recall, medium_object_tracker_mean_recall, small_object_tracker_mean_recall, mean_iou, tracker_mean_iou, Image_size, tracker_type_str, F1_score_str])
        print(results)
        f.write(results + '\n')

def accu_iou_comput(history, labels, iou_tresh):
    iou_history = []
    TPs = []
    FNs = []
    FPs = []

    small_object_TPs = []
    small_object_FNs = []
    small_object_FPs = []

    medium_object_TPs = []
    medium_object_FNs = []
    medium_object_FPs = []

    big_object_TPs = []
    big_object_FNs = []
    big_object_FPs = []

    num_small = 0
    num_medium = 0
    num_big = 0

    labels_sum = 0

    for i, bbox in enumerate(history):
        if i >= len(labels):
            break
        boxes = bbox
        indexes = []
        ious = []
        TP = 0
        FN = len(labels[i])
        FP = max(0, len(boxes) - len(labels[i]))

        small_object_TP = 0
        small_object_FN = 0
        small_object_FP = 0

        medium_object_TP = 0
        medium_object_FN = 0
        medium_object_FP = 0

        big_object_TP = 0
        big_object_FN = 0
        big_object_FP = 0

        for box in bbox:

            area_pred = (box[3] - box[1]) * (box[2] - box[0])

            if area_pred >= 0.25:
                big_object_FP += 1
            elif area_pred < 0.25 and area_pred >= (0.25 * 0.25):
                medium_object_FP += 1
            elif area_pred < (0.25 * 0.25):
                small_object_FP += 1

        for label in labels[i]:
            area = (label[3]-label[1]) * (label[2]-label[0])

            if area >= 0.25:
                num_big += 1
                big_object_FN += 1
            elif area < 0.25 and area >= (0.25 * 0.25):
                num_medium += 1
                medium_object_FN += 1
            elif area < (0.25 * 0.25):
                num_small += 1
                small_object_FN += 1

            inter = [intersection_over_union(np.array(box), np.array(label)) for box in bbox]

            # Check if inter is not empty and contains non-empty arrays
            if inter and any(arr.size != 0 for arr in inter):
                # Get the maximum IoU value
                iou_box = np.max(inter)
                # Get the index of the maximum IoU value
                max_index = np.argmax(inter)

                if iou_box >= iou_tresh:
                    TP += 1
                    FN = max(0, FN - 1)

                if area >= 0.25 and iou_box >= iou_tresh:
                        big_object_TP += 1
                        big_object_FN -= 1
                        big_object_FP -= 1
                elif area < 0.25 and area >= (0.25 * 0.25) and iou_box >= iou_tresh:
                        medium_object_TP += 1
                        medium_object_FN -= 1
                        medium_object_FP -= 1
                elif area < (0.25 * 0.25) and iou_box >= iou_tresh:
                        small_object_TP += 1
                        small_object_FN -= 1
                        small_object_FP -= 1

                # Remove the bounding box with the maximum IoU from bbox
                del bbox[max_index]

            else:
                iou_box = 0
                max_index = None  # or -1 or any other value indicating no valid index
                # print('No detection or intersection')


            big_object_FP = max(0, big_object_FP)
            big_object_FN = max(0, big_object_FN)

            medium_object_FP = max(0, medium_object_FP)
            medium_object_FN = max(0, medium_object_FN)

            small_object_FP = max(0, small_object_FP)
            small_object_FN = max(0, small_object_FN)

        indexes.append(max_index)
        ious.append(iou_box)
        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)

        big_object_TPs.append(big_object_TP)
        big_object_FPs.append(big_object_FP)
        big_object_FNs.append(big_object_FN)

        medium_object_TPs.append(medium_object_TP)
        medium_object_FPs.append(medium_object_FP)
        medium_object_FNs.append(medium_object_FN)

        small_object_TPs.append(small_object_TP)
        small_object_FPs.append(small_object_FP)
        small_object_FNs.append(small_object_FN)

        iou_history.append(np.sum(ious)/len(ious))

    print(f'{num_big, num_medium, num_small = }')

    return iou_history, TPs, FPs, FNs, big_object_TPs, big_object_FPs, big_object_FNs, medium_object_TPs, medium_object_FPs, medium_object_FNs, small_object_TPs, small_object_FPs, small_object_FNs
