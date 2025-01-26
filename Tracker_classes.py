import sys
import numpy as np
from itertools import product
from typing import List, Tuple
import cv2
import torch
from random import randrange
from utils.SFSORT import SFSORT
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    """Multi-Object Tracking System"""

    def __init__(self, tracker_type, confidenceThreshold, frame_rate, frame_width, frame_height, detections_init = None, frame_init = None):
        """Initialize a tracker with given arguments"""
        self.tracker_type = tracker_type
        self.frame_rate = frame_rate
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.confidenceThreshold = confidenceThreshold

        self.tracker_SFSORT_arguments = {"dynamic_tuning": True, "cth": 0.7,
                                    "high_th": 0.82, "high_th_m": 0.1,
                                    "match_th_first": 0.5, "match_th_first_m": 0.05,
                                    "match_th_second": 0.1, "low_th": 0.3,
                                    "new_track_th": 0.7, "new_track_th_m": 0.1,
                                    "marginal_timeout": (7 * self.frame_rate // 10),
                                    "central_timeout": self.frame_rate,
                                    "horizontal_margin": self.frame_width // 10,
                                    "vertical_margin": self.frame_height // 10,
                                    "frame_width": self.frame_width,
                                    "frame_height": self.frame_height}


        self.basic_trackers = {

            'csrt': cv2.legacy.TrackerCSRT_create,  # high accuracy ,slow
            'mosse': cv2.legacy.TrackerMOSSE_create,  # fast, low accuracy
            'kcf': cv2.legacy.TrackerKCF_create,  # moderate accuracy and speed
            'medianflow': cv2.legacy.TrackerMedianFlow_create,
            'mil': cv2.legacy.TrackerMIL_create,
            'tld': cv2.legacy.TrackerTLD_create,
            'boosting': cv2.legacy.TrackerBoosting_create
        }


        if self.tracker_type != 'SFSORT' and self.tracker_type != 'DEEPSORT':
            # print('Basic tracker created')
            self.tracker = self.multi_tracker_init(frame_init, detections_init, tracker_type)
            self.tracker_fn = self.basic_tracker_fn
        elif self.tracker_type == 'SFSORT':
            # print('SFSORT tracker created')
            self.tracker = SFSORT(self.tracker_SFSORT_arguments)
            self.tracker_fn = self.SFSORT_fn
        elif self.tracker_type == 'DEEPSORT':
            # print('SFSORT tracker created')
            self.tracker = DeepSort(max_age=5)
            self.tracker_fn = self.DEEPSORT_fn
        else:
            print('Unknown tracker passed')



    def multi_tracker_init(self, frame, detections, tracker_type):
        # Create MultiTracker object
        multiTracker = cv2.legacy.MultiTracker_create()
        self.id_stor = []
        self.class_stor = []
        self.lost_it = []

        for key, value in detections.items():
            # tracker_unit = self.trackers[tracker_type]()
            bbox = np.array([value[0][0], value[0][1], value[0][2]-value[0][0], value[0][3]-value[0][1]]).astype(int)
            # print(f'init with {bbox = }')
            multiTracker.add(self.basic_trackers[self.tracker_type](), frame, bbox)
            # multiTracker.add(self.createTrackerByName(self.tracker_type), frame, bbox)
            self.id_stor.append(key)
            self.lost_it.append(value[2])
            self.class_stor.append(int(value[0][5]))
        return multiTracker

    def basic_tracker_fn(self, frame):
        out_track = {}
        # if frame is not None:
        success, bboxes = self.tracker.update(frame)  ## return numpy array

        for i, box in enumerate(bboxes):
            newbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]] ## should be that format for all basic trackers (x,y,x+w,y+h) --from xywh to xyxy
            # newbox = [box[0], box[1], box[2], box[3]]
            box_out = np.append(newbox, [0.5, self.class_stor[i]])

            out_track[self.id_stor[i]] = [box_out, 'tracked', self.lost_it[i] + 1]

        return out_track


    def clear(self):
        self.tracker.clear()

    def SFSORT_fn(self, detections, frame):
        results = self.convert_detections_SFSORT(detections, classes=0)
        prediction_results_xyxy, prediction_results_conf = np.array([result[0] for result in results]), np.array(
            [result[1] for result in results])
        tracks = self.tracker.update(prediction_results_xyxy, prediction_results_conf)
        detections = self.convert_back_from_SFSORT(tracks)
        return detections

    def DEEPSORT_fn(self, detections, frame):
        results = self.convert_detections_DEEPSORT(detections)

        tracks = self.tracker.update_tracks(results, frame=frame)

        detections = self.convert_back_from_DEEPSORT(tracks)

        return detections

    # Define a function to convert detections to SORT format.
    def convert_detections_SFSORT(self, detections, classes=0):
        # Get the bounding boxes, labels and scores from the detections dictionary.
        boxes = []
        labels = []
        scores = []

        for k, v in detections.items():
            boxes.append(v[0][0:4])
            labels.append(v[0][5])
            scores.append(v[0][4])

        boxes = np.array(boxes).astype(int)
        labels = np.array(labels)
        scores = np.array(scores)

        # Convert boxes to [x1, y1, w, h, score] format.
        final_boxes = []
        for i, box in enumerate(boxes):
            # Append ([x, y, x, y], score, label_string).
            final_boxes.append(
                (
                    [box[0], box[1], box[2] , box[3]],
                    scores[i],
                    str(labels[i])
                )
            )

        return final_boxes

    # Define a function to convert detections to SORT format.
    def convert_detections_DEEPSORT(self, detections, classes=0):
        # Get the bounding boxes, labels and scores from the detections dictionary.
        boxes = []
        labels = []
        scores = []

        for k, v in detections.items():
            boxes.append(v[0][0:4])
            labels.append(v[0][5])
            scores.append(v[0][4])

        boxes = np.array(boxes).astype(int)
        labels = np.array(labels)
        scores = np.array(scores)

        # Convert boxes to [x1, y1, w, h, score] format.
        final_boxes = []
        for i, box in enumerate(boxes):
            # Append ([x, y, x, y], score, label_string).
            final_boxes.append(
                (
                    [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    scores[i],
                    str(labels[i])
                )
            )

        return final_boxes

    def convert_back_from_SFSORT(self, tracks):
        detec = {}
        for track in tracks:
            bbox = np.array(track[0], dtype=float)
            bbox = np.append(bbox, [0.5, 0])
            detec[f'{int(track[1])}'] = [bbox, 'tracked', 0]

        return detec

    def convert_back_from_DEEPSORT(self, tracks):
        detec = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            track_class = float(track.det_class)
            bbox = np.array(track.to_ltrb(), dtype=float)
            bbox = np.append(bbox, [0.5, track_class])
            detec[f'id_{track_id}'] = [bbox, 'tracked', 0]

        return detec

