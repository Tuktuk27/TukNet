#!/usr/bin/env python3
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter
from utils.utils import decode_nms, displayFrame_tracker
from utils.Tracker import Tracker


# model_name = 'yolo_u1head_shuffle_224' ## 66 or 93 ?? fps ||
model_name = 'yolo_u1head_shuffle_relu_416_5_resize' ## 66 fps ||
# model_name = 'yolo_u1head_shuffle_relu_320_no-resize' ## 66 fps ||



# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default=f'model/{model_name}.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='json/yolox_224.json', type=str)
parser.add_argument("-topk", "--keep_top_k", default=750, type=int, help='set keep_top_k for results outputing.')
parser.add_argument("-iou_track", "--iou_tracking", default=0.3, type=float, help='set the iou for assigning id.')
parser.add_argument("-tracktype", "--tracker_type", default='SFSORT', type=str, help='Tracking algorithm.')
parser.add_argument("-track", "--to_be_tracked", default=True, type=bool, help='Choose to track or not.')

args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    NN_WIDTH, NN_HEIGHT = tuple(map(int, nnConfig.get("input_size").split('x')))
else:
    NN_WIDTH, NN_HEIGHT = 256, 256
NN_WIDTH, NN_HEIGHT = 416, 416
# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.MonoCamera)
detectionNetwork = pipeline.create(dai.node.NeuralNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRgb.setFps(120)

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
manip.inputConfig.setWaitForMessage(False)

script = pipeline.createScript()

xin = pipeline.create(dai.node.XLinkIn)
xin.setStreamName('in')
xin.out.link(script.inputs['toggle'])

manip.out.link(script.inputs['rgb'])
script.setScript("""
    toggle = True
    while True:
        msg = node.io['toggle'].tryGet()
        if msg is not None:
            toggle = msg.getData()[0]
            node.warn('Toggle! Perform NN inferencing: ' + str(toggle))

        frame = node.io['rgb'].get()

        if toggle:
            node.io['nn'].send(frame)
""")

script.outputs['nn'].link(detectionNetwork.input)

# Network specific settings
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# # Linking
camRgb.out.link(manip.inputImage)
# manip.out.link(detectionNetwork.input)
# detectionNetwork.passthrough.link(xoutRgb.input)
detectionNetwork.out.link(nnOut.input)

manip.out.link(xoutRgb.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    ## Debugging and device state
    # device.setLogLevel(dai.LogLevel.DEBUG)
    # device.setLogOutputLevel(dai.LogLevel.DEBUG)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    inQ = device.getInputQueue("in")
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    runNn = True

    frame = None
    detections = []
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.perf_counter()
    fps_smoothed = None
    fps_alpha = 0.2  # Smoothing factor (adjust as needed)
    fps = 0

    tracking = False
    reset_tracking_counter = 0

    if (args.tracker_type == 'DEEPSORT' or args.tracker_type == 'SFSORT') and args.to_be_tracked:
        tracker = Tracker(args.tracker_type, confidenceThreshold, frame_rate=60, frame_width=NN_WIDTH,
                              frame_height=NN_HEIGHT)
        tracking = True

    while True:
        inRgb = qRgb.get()

        detections = {}
        out = {}
        
        # Increment frame count
        frame_count += 1

        # Check if a second has elapsed
        current_time = time.perf_counter()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1.0:
            # Calculate FPS
            fps = frame_count / elapsed_time

            # Smooth FPS using exponential moving average
            if fps_smoothed is None:
                fps_smoothed = fps
            else:
                fps_smoothed = fps_alpha * fps + (1 - fps_alpha) * fps_smoothed

            # Reset variables for next FPS calculation
            start_time = current_time
            frame_count = 0
            reset_tracking_counter += 1

        if inRgb is not None:
            frame = inRgb.getCvFrame()

            cv2.putText(frame, "NN fps: {:.2f}".format(fps),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))

            print(f'{fps = }')

        if qDet.has():
            inDet = qDet.get()

            detections = decode_nms((NN_WIDTH, NN_HEIGHT), inDet.getLayerFp16("output"), args.keep_top_k,
                                confidenceThreshold, iouThreshold, u1x = False)

        detections_bbox, out = displayFrame_tracker(frame, detections, model_name, labels, fps)

        if args.to_be_tracked and (args.tracker_type =='DEEPSORT' or args.tracker_type =='SFSORT'):
            trackeds = tracker.tracker_fn(detections, frame)
        else:
            if not tracking and args.to_be_tracked:
                # out = {'0': [np.array([5, 6, 200, 250, 0.53572678565979, 0]), 'detected', 0]}
                if len(out) != 0:
                    start = time.monotonic()
                    tracker = Tracker(args.tracker_type, confidenceThreshold, frame_rate=60, frame_width=NN_WIDTH,
                                      frame_height=NN_HEIGHT, detections_init = out, frame_init = frame)
                    print(f'Time required to instantiate the tracker {time.monotonic() - start = }')
                    tracking = True
                    trackeds = out
                    runNn = not runNn
                    print(f"{'Enabling' if runNn else 'Disabling'} NN inferencing")
                    buf = dai.Buffer()
                    buf.setData(runNn)
                    inQ.send(buf)
            elif tracking and args.to_be_tracked:
                trackeds = tracker.tracker_fn(frame)

        if tracking and args.to_be_tracked:
            tracks, _ = displayFrame_tracker(frame, trackeds, 'tracking', labels, fps)
            if reset_tracking_counter >= 15 and not (args.tracker_type =='DEEPSORT' or args.tracker_type =='SFSORT'):
                tracking = False
                tracker.clear()
                del tracker
                reset_tracking_counter = 0
                runNn = not runNn
                print(f"{'Enabling' if runNn else 'Disabling'} NN inferencing")
                buf = dai.Buffer()
                buf.setData(runNn)
                inQ.send(buf)

            

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('t'):
            runNn = not runNn
            print(f"{'Enabling' if runNn else 'Disabling'} NN inferencing")
            buf = dai.Buffer()
            buf.setData(runNn)
            inQ.send(buf)
