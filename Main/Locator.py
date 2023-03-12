import os
import cv2
import urllib.request

# Download the cascade classifier file
url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
urllib.request.urlretrieve(url, 'cascade.xml')

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('cascade.xml')

# Capture the video from the default camera
cap = cv2.VideoCapture(0)

# Set the path where the screenshots will be saved
folder_path = os.path.dirname(os.path.abspath(__file__))

# Initialize a counter for the screenshot filename
count = 0

while True:
    # Read the frame
    ret, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save the screenshot if a face is detected
        filename = os.path.join(folder_path, f'screenshot{count}.png')
        cv2.imwrite(filename, img)
        count += 1

        # Move the mouse cursor to the center of the face
        center_x = x + w // 2
        center_y = y + h // 2
        # Uncomment the line below if you want to move the mouse cursor
        # pyautogui.moveTo(center_x, center_y)

    # Display the output
    cv2.imshow('img', img)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

    # Add a delay to limit the frame rate and prevent excessive resource usage
    cv2.waitKey(5000)

# Release the resources
cap.release()
cv2.destroyAllWindows()
