# Swift Object Recognition and Tracking for fast Drones using High-Speed Cameras and Deep Learning

# Abstract

This master’s thesis aims to rapidly identify, and comprehend objects of interest by combining high
speed cameras, specifically Luxonis OAK-D Lite camera, with customed, tiny and efficient Pytorch 
models, with the final goal of implementation on fast drones (fast-speed drones). The objective is to 
achieve real-time (using a camera on board) and offline (frame-by-frame on video) implementations. 
The significance of this project lies in its comprehensive approach, integrating fast object detection, 
along with tracking. The expected outcome is a flexible and versatile system implemented on Luxonis 
OAK Camera and designed to meet the speed requirements for fast drones detection. 
This master’s thesis explores the realms of computer vision and machine learning, uniting hardware 
and software to achieve swift and intelligent object recognition and dynamic tracking. 


# YOLO Models:

- YOLOv8 and YOLOX: Balance of speed and accuracy, Anchor-free detection and adaptive training sample selection

Lightweight Models for Edge Devices:

- ShuffleNet: Optimizes speed and performance on mobile and edge devices
- YuNet: Real-time face detector with only 75 856 parameters

# Architecture:

ShuffleNet backbone :

<img src="https://github.com/user-attachments/assets/857d40fd-62f5-4cf3-964d-4f68f28b216f" alt="Sample Image" width="600">

YOLOX head:

<img src="https://github.com/user-attachments/assets/a9d5a2b5-49ec-47d9-9c20-dd5b42484ccc" alt="Sample Image" width="600">

Yunet head:

<img src="https://github.com/user-attachments/assets/f2e0cb02-87b1-4b16-afbb-60cec18492ca" alt="Sample Image" width="300">

Architecture used:

- TukNetv1: ShuffleNet backbone, customed neck, and YuNet head  
- TukNetv2: ShuffleNet backbone, custom neck, and YOLOX head 

# Results:

<img src="https://github.com/user-attachments/assets/1782d768-67bb-4629-9cdc-0ef978573d3b" alt="Sample Image" width="600">

<img src="https://github.com/user-attachments/assets/5a4e6eea-65a4-47be-a2a2-83114922354a" alt="Sample Image" width="600">

<img src="https://github.com/user-attachments/assets/c57fd226-1c0b-46ca-9a69-77f376f8239c" alt="Sample Image">


![image](https://github.com/user-attachments/assets/48cf1ae5-3d64-49f0-8b8e-9e7a689d990c)

![image](https://github.com/user-attachments/assets/f2ffa474-c450-4b91-b785-21945665169a)


# CONCLUSION

The results of this study demonstrate that the object detection model successfully met the real-time 
requirements, achieving performance levels exceeding 100 FPS. The designed models were more 
efficient than Yolov8n and Yolox and this has pointed out the importance of the model architecture to 
target specific scales for object detection, but also to target specific hardware requirement such as 
speed given a certain computational environment. Future efforts could focus on further optimizing the 
models, however, to develop a comprehensive, swift, and reliable system specifically optimized for 
tracking fast drones, it seems more important to incorporate pose estimation to these models. 
Although computational power was limited, it did not ultimately impede our real-time system, and 
Luxonis OAK modules showed promise for this application. Color cameras should however be used in 
order to improve the reliability and versatility of the system. 
Looking ahead, the other challenges will also involve extending the maximum detection distance and 
developing hybrid systems tailored to two different scenarios: First detection on a wide surrounding, 
and then tracking on a small and specific area or it. Combining neural network-based detection with 
traditional tracking methods appears to be a practical and effective approach, but not in the manner 
thought of at first. The strategy to create a versatile and efficient monitoring hybrid system, would 
rather use traditional tracking in the initial detection stage for motion flow tracking, while neural 
network models would later enhance speed and precision on a relevant and specific object or area. 
Overall, this project has demonstrated the feasibility of real-time drone detection and tracking using 
advanced neural network models. It sets a foundation for further advancements and the integration 
of more sophisticated features such as pose estimation and hybrid detection systems.

# Credits
- Yunet: https://github.com/geaxgx/depthai_yunet
- ShuffleNet: Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." Proceedings of the European conference on computer vision (ECCV). 2018
- YoloX: Ge, Zheng & Liu, Songtao & Wang, Feng & Li, Zeming & Sun, Jian. (2021). YOLOX: Exceeding YOLO Series in 2021
