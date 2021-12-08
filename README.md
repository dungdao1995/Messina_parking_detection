# Messina Parking Slot Detection Using Mask R-CNN model

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample](assets/street.png)

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset

# Motivation
The search for a parking space in urban areas is often time-consuming and nerve-racking. Ef- ficient car park guidance systems could support drivers in their search for an available parking space. Video-based systems are a reasonably priced alternative to systems employing other sensor types and their camera input can be used for various tasks within the system.
Current systems detecting vacant parking spaces are either very expensive due to their hardware requirements or do not provide a detailed occupancy map. While several sensor types feature individual parking space surveillance, their installation and maintenance costs are rela- tively high. The system developed in this research group has minimal hardware requirements, which makes it less expensive and easy to install. At the same time, our video-based approach offers flexibility regarding information usage and site of operation.

# Project Overview
• Using a camera to collect real-time data in parking space and process video into images.
• Using the Deep Learning Model (Computer Vision Technique) to detect parking slots
which is deployed on local machine.
• When detecting a free parking slot, sending result to the user’s interface.
• User’s interface are Moblie Apps or Web Apps.
