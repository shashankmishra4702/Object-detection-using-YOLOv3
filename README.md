# Object-detection-using-YOLOv3
YOLOv3 (You Only Look Once, Version 3) is a state-of-the-art, real-time object detection system. It is known for its speed and accuracy, making it popular in various applications such as autonomous driving, surveillance, and robotics.

Architecture: YOLOv3 uses a deep convolutional neural network (CNN) to predict bounding boxes and class probabilities directly from full images in a single evaluation.
Grid System: The image is divided into an S×S grid. Each grid cell is responsible for detecting objects whose centers fall within the cell.
Bounding Boxes: For each grid cell, YOLO predicts multiple bounding boxes, each with a confidence score. The confidence score reflects the accuracy of the bounding box and whether it contains an object.
Class Prediction: Each bounding box has associated class probabilities, which indicate the likelihood of the object belonging to each class.

This project uses YOLO coco dataset which can be downloaded from this link 
https://www.kaggle.com/datasets/valentynsichkar/yolo-coco-data

The results of the model can be shown as -->

Image 1

![image](https://github.com/comdershashank/Object-detection-using-YOLOv3/assets/105141896/f216be9f-a63e-4ccf-8301-f5107d67c0f9)

Output 1

![image](https://github.com/comdershashank/Object-detection-using-YOLOv3/assets/105141896/7f692dac-570d-44d8-beb6-f59c966a0c51)

Image 2

![image](https://github.com/comdershashank/Object-detection-using-YOLOv3/assets/105141896/57f2e218-4a60-4ae5-a5fa-39c34a2ad98c)

Output 2

![image](https://github.com/comdershashank/Object-detection-using-YOLOv3/assets/105141896/45fdef47-6f0a-4eb0-a4a2-00fc69e01f86)



