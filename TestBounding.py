import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
from pathlib import Path

# The function to get the bounding of Slots
def get_bboxSlots():
    myTree = ET.parse('videos/parkingSlot_test1.xml')
    myRoot = myTree.getroot()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for i in myRoot.findall('object'):
        xmin.append(int(i.find('bndbox').find('xmin').text))
        ymin.append(int(i.find('bndbox').find('ymin').text))
        xmax.append(int(i.find('bndbox').find('xmax').text))
        ymax.append(int(i.find('bndbox').find('ymax').text))

    matrix_slots = np.array([ymin,xmin,ymax,xmax])
    parking_slots = np.array(matrix_slots.transpose())

    return parking_slots

ROOT_DIR = Path(".")

IMAGE_DIR = os.path.join(ROOT_DIR, "videos/0.jpg")

#Cut the first frame to draw bbox
VIDEO_SOURCE = "videos/Khare_testvideo_03.mp4"
video_capture = cv2.VideoCapture(VIDEO_SOURCE)
def save_Frame():
    count = 0
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        if count ==0:
            name = str(count) + ".jpg"
            name = os.path.join('./videos', name)
            cv2.imwrite(name, frame)
        count+=1

#Draw the boundingbox
def bounding_Slots(parking_slots,IMAGE_DIR):
    # Read input image
    img = cv2.imread(IMAGE_DIR)
    for i in parking_slots:
        y1, x1, y2, x2 = i
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Image",img)
    cv2.waitKey(0)

if __name__ == "__main__":
    parking_slots = get_bboxSlots()
    print(parking_slots)
    bounding_Slots(parking_slots,IMAGE_DIR)