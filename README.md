Swift Object Recognition and Tracking for fast Drones using High-Speed Cameras and Deep Learning
YOLO Models:
YOLOv8 and YOLOX: Balance of speed and accuracy
Anchor-free detection and adaptive training sample selection
Lightweight Models for Edge Devices:
ShuffleNet: Optimizes speed and performance on mobile and edge devices
YuNet --> Real-time face detector with only 75 856 parameters

TukNet:
ShuffleNet backbone :
![image](https://github.com/user-attachments/assets/857d40fd-62f5-4cf3-964d-4f68f28b216f)


YOLOX head:
![image](https://github.com/user-attachments/assets/a9d5a2b5-49ec-47d9-9c20-dd5b42484ccc)

Yunet head:
![image](https://github.com/user-attachments/assets/f2e0cb02-87b1-4b16-afbb-60cec18492ca)


TukNetv1: ShuffleNet backbone, customed neck, and YuNet head  
TukNetv2: ShuffleNet backbone, custom neck, and YOLOX head 

![image](https://github.com/user-attachments/assets/c57fd226-1c0b-46ca-9a69-77f376f8239c)


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


