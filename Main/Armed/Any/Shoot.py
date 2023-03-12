import cv2
import time
import serial
import random
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Define the range of HSV values for the color of the gun
lower_color = np.array([110,50,50])
upper_color = np.array([130,255,255])

# Define the serial port for communication with the Arduino
ser = serial.Serial('COM3', 9600)  # Replace with the serial port of your Arduino board

# Define the angles for aiming the gun
left_angle = 90
right_angle = 180

# Initialize the last detected face position as None
last_face_pos = None

while True:
    # Read a frame from the camera
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save the position of the last detected face
        last_face_pos = (x, y, w, h)

    # If a face was detected in the current frame
    if last_face_pos is not None:
        # Crop the frame to the region of interest (ROI) around the face
        x, y, w, h = last_face_pos
        roi = frame[y:y+h, x:x+w]

        # Convert the ROI to the HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Threshold the image to extract the gun
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find the contours of the gun in the thresholded image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If contours were found
        if len(contours) > 0:
            # Find the contour with the largest area (the gun)
            gun_contour = max(contours, key=cv2.contourArea)

            # Find the bounding rectangle of the gun contour
            x_gun, y_gun, w_gun, h_gun = cv2.boundingRect(gun_contour)

            # Calculate the center of the gun
            cx_gun = x_gun + w_gun//2
            cy_gun = y_gun + h_gun//2

            # Calculate the error (difference) between the center of the gun and the center of the ROI
            error = cx_gun - w//2

            # Calculate the angle to aim the gun
            if error < 0:
                angle = left_angle
            else:
                angle = right_angle

            # Send the angle to the Arduino to move the servo
            ser.write(str(angle).encode())

            # Wait for the servo to move to the desired angle
            time.sleep(1)

            # Trigger the gun by sending a signal to the Arduino
            ser.write(b'1')

            # Wait for the gun to shoot
            time.sleep(1)

    # Show the frame with the detected faces and gun
    cv2.imshow('frame', frame)

    # Wait for a key
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Exit on ESC
        break
    elif k == ord(' '):  # Trigger gun on SPACE key
        ser.write(b'1')
        time.sleep(1)

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
