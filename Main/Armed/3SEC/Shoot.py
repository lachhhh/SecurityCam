import cv2
import time

# Load the face detection cascade classifier
face_cascade = cv2.CascadeClassifier('cascade.xml')

# Initialize the video stream
cap = cv2.VideoCapture(0)

# Get the width and height of the video stream
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the position of the dot and the square
dot_pos = (int(width/2), int(height/2))
square_pos = None

# Set the delay time for shooting
shoot_delay = 3  # in seconds
shoot_time = None

# Continuously capture frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Find the position of the square
    square_pos = None
    for (x, y, w, h) in faces:
        # Save the position of the square
        square_pos = (x, y, w, h)
        # Draw a green square around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw the dot on the frame
    cv2.circle(frame, dot_pos, 5, (0, 0, 255), -1)

    # Check for collision between the dot and the square
    if square_pos is not None:
        x, y, w, h = square_pos
        if dot_pos[0] >= x and dot_pos[0] <= x + w and dot_pos[1] >= y and dot_pos[1] <= y + h:
            # Record the shooting time if it hasn't been recorded yet
            if shoot_time is None:
                shoot_time = time.time()
            # Check if the shooting delay has passed
            if time.time() - shoot_time >= shoot_delay:
                print("shoot")
                # Reset the shooting time and the dot position
                shoot_time = None
                dot_pos = (int(width/2), int(height/2))
        else:
            # Reset the shooting time if there's no collision
            shoot_time = None

    # Update the position of the dot based on the user input
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    

    # Display the resulting frame
    cv2.imshow('frame', frame)

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
