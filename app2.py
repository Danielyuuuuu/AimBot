import cv2
import numpy as np
from PIL import ImageGrab
import keyboard
import time
import threading
import random
from pynput import mouse, keyboard
from pynput.keyboard import KeyCode
import sys
import os

# Load the pre-trained human detection model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the list of classes (COCO dataset)
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Set the threshold for confidence in detections
conf_threshold = 0.5

# Set the threshold for non-maximum suppression
nms_threshold = 0.4

def detect_humans(image):
    # Perform forward pass through the network
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Process each output layer
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions
            if confidence > conf_threshold and class_id == 0:  # Class ID 0 corresponds to 'person'
                # Scale the bounding box coordinates to the image size
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, bbox_width, bbox_height = box.astype("int")

                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - (bbox_width / 2))
                y = int(center_y - (bbox_height / 2))

                # Add the coordinates, confidence, and class ID to the respective lists
                boxes.append([x, y, int(bbox_width), int(bbox_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Initialize the list of coordinates
    coordinates = []

    # Process the remaining bounding boxes after non-maximum suppression
    for i in range(len(boxes)):
        if i in indices:
            # Extract the coordinates of the bounding box
            x, y, w, h = boxes[i]

            # Add the top-left corner coordinates and dimensions to the list
            coordinates.append((x, y, w, h))

    return coordinates


# Capture the screen continuously until 'q' is pressed
def pattern_matching():
    while True:
        # Capture the screen
        screen = np.array(ImageGrab.grab(bbox=(0, 0, 2560, 1440)))  # Adjust the bounding box as per your screen resolution

        # Detect humans in the screen capture
        human_coordinates = detect_humans(screen)

        # Print the coordinates of the detected humans
        for coord in human_coordinates:
            print("Human detected at coordinates:", coord)

        # Display the screen capture with bounding boxes
        for (x, y, w, h) in human_coordinates:
            cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # cv2.imshow('Screen Capture', screen)

        time.sleep(1)

        # # Check if the 'q' key is pressed to exit the program
        # if keyboard.is_pressed('q'):  # if key 'q' is pressed 
        #     break

def keyboard_press(key):
    # def on_press(key):
    if key == KeyCode.from_char('1'):
        print('Exit')
        os._exit(0)
    
    # with keyboard.Listener(on_press=on_press) as listener:
    #     listener.join()

def main():
    threading.Thread(target=pattern_matching).start()
    # threading.Thread(target=keyboard_press).start()

    with keyboard.Listener(on_press=keyboard_press) as listener:
        listener.join()

    # Release the capture and close the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()