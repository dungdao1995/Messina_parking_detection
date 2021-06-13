import tensorflow
#import
import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import certifi

from flask import Flask, render_template, request
import pymongo
import dns

import xml.etree.ElementTree as ET


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

parking_slots = get_bboxSlots()
Slots = {}
Slots_free_space = {}
Slots_free_space_frames = {}
Slots_status = {}
for x,y in enumerate(parking_slots):
    Slots['Slot '+str(x+1)] = y
    Slots_free_space['Slot '+str(x+1)] = False
    Slots_free_space_frames['Slot '+str(x+1)] = 0
    Slots_status['Slot '+str(x+1)] = 'Parked'



#Get the Key from Value
def get_key(my_dict,val):
    for key, value in my_dict.items():
        if (val == value).all():
            return key

    return "key doesn't exist"

# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Video file or camera to process - set this to 0 to use your webcam instead of a video file
VIDEO_SOURCE = "videos/Khare_testvideo_03.mp4"

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
parked_car_boxes = None

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# How many frames of video we've seen in a row with a parking space open
free_space_frames = 0

# Have we sent an SMS alert yet?
sms_sent = False


count = 0

#Total slots
total_slots = len(Slots_status)
#Total_available slots
available_slots = 0

#Server
client = pymongo.MongoClient("mongodb+srv://admin:admin@cluster0.riqct.mongodb.net/myFirstDatabase?retryWrites=true&w=majority",tlsCAFile=certifi.where())
db = client.get_database('parking_db_3')
records = db.parking_records

if __name__ == "__main__":
    # Loop over each frame of video
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_image = frame[:, :, ::-1]

        # Run the image through the Mask R-CNN model to get results.
        results = model.detect([rgb_image], verbose=0)

        # Mask R-CNN assumes we are running detection on multiple images.
        # We only passed in one image to detect, so only grab the first result.
        r = results[0]

        # The r variable will now have the results of detection:
        # - r['rois'] are the bounding box of each detected object
        # - r['class_ids'] are the class id (type) of each detected object
        # - r['scores'] are the confidence scores for each detection
        # - r['masks'] are the object masks for each detected object (which gives you the object outline)

        # This is the first frame of video - assume all the cars detected are in parking spaces.
        # Save the location of each car as a parking space box and go to the next frame of video.
        parked_car_boxes = get_bboxSlots()
        # print("Bounding box slots: ", parked_car_boxes)
        # print("Bounding box slots shape: ", parked_car_boxes.shape)


        # Get where cars are currently located in the frame
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        # print("Bounding car detected: ",car_boxes)
        # print("Bounding car detected shape: ",car_boxes.shape)

        if len(car_boxes) !=0:
            # We already know where the parking spaces are. Check if any are currently unoccupied.

            # See how much those cars overlap with the known parking spaces
            overlaps = mrcnn.utils.compute_overlaps(parked_car_boxes, car_boxes)

            # Assume no spaces are free until we find one that is free
            #free_space = False
            free_space = {}
            for x,y in enumerate(parking_slots):
                free_space['Slot '+str(x+1)] = False
            #print("Free Spcae must be reset 0 in each Frame", free_space)
            # Loop through each known parking space box
            for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):

                # For this parking space, find the max amount it was covered by any
                # car that was detected in our image (doesn't really matter which car)
                max_IoU_overlap = np.max(overlap_areas)

                # Get the top-left and bottom-right coordinates of the parking area
                y1, x1, y2, x2 = parking_area

                #Get the slot
                slot_name = get_key(Slots,parking_area)

                # Check if the parking space is occupied by seeing if any car overlaps
                # it by more than 0.2 using IoU
                if max_IoU_overlap < 0.2:
                    # Parking space not occupied! Draw a green box around it
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Flag that we have seen at least one open space
                    #free_space = True
                    free_space[slot_name] = True
                else:
                    # Parking space is still occupied - draw a red box around it
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

                # Write the IoU measurement inside the box
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))

                #Print the Slot_name
                cv2.putText(frame, f"{slot_name}", (x1 + 6, y1 + 6), font, 0.3, (255, 255, 255))
            #print("FreeSpace after detecting", free_space)

            #print("Counting frame in each slot: ", Slots_free_space_frames)
            for key,value in free_space.items():
                if value == True:
                    Slots_free_space_frames[key] += 1
                else:
                    # If no spots are free, reset the count
                    Slots_free_space_frames[key] = 0
                    Slots_status[key] = 'Parked'
            # If a space has been free for several frames, we are pretty sure it is really free!
            for key, value in Slots_free_space_frames.items():
                space = int(key[5:])
                #we have 15 frames/s, so 60 frames equal 4s
                if value > 10:
                    #Update final status
                    Slots_status[key] = "Available"
                    # Write SPACE AVAILABLE!! at the top of the screen
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, key+ f" available!", (10, space*10), font, 0.4, (0, 255, 0))

            # #saving each frame
            # name = str(count) + ".jpg"
            # name = os.path.join('/content/drive/Shareddrives/DKD/Distributed_System/Result', name)
            # cv2.imwrite(name, frame)

            if count%1 ==0:
                print(Slots_status)
                available_slots = sum(value == 'Available' for value in Slots_status.values())
                print('Available:', available_slots)
                print('Total slots: ', total_slots )
                #Sending Data
                records.update_one(filter={'total':total_slots}, update={'$set':{'available' :available_slots }})
                for i in range(len(Slots_status)):
                    records.update_one(filter={'order':i+1}, update={'$set':{'status' :Slots_status[list(Slots_status)[i]] }})

            # Show the frame of video on the screen
            #cv2_imshow(frame)

            #Count +=1
            count += 1
        #No car in the Parking Space
        else:
            for parking_area in parked_car_boxes:
                y1, x1, y2, x2 = parking_area
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            for key, value in Slots_free_space_frames.items():
                # Write SPACE AVAILABLE!! at the top of the screen
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, key+ f" available!", (10, space*10), font, 0.4, (0, 255, 0))
                #Update the final status
                Slots_status[key] = "Available"

            print(Slots_status)
            cv2.imshow(frame)

        # Hit 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up everything when finished
    video_capture.release()
    #v2.destroyAllWindows())
