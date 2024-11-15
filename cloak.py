import cv2   # Computer vision
import numpy as np
import time

# Start your default laptop camera (index 0 for default, 1 or 2 if not working)
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(3)  # Time to open the camera

# Capture background for later use
for i in range(60):  # Time to store image
    check, background = video.read()  # Use loop to get precise image
background = np.flip(background, axis=1)  # Flip image horizontally

while video.isOpened():
    check, img = video.read()
    if not check:
        break  # Checking if the camera feed is working
    img = np.flip(img, axis=1)  # Flip the image horizontally to match normal view

    # Convert image from BGR to HSV (better color detection)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the color range for red (can be adjusted for other colors)
    lower_red = np.array([0, 120, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])  # Range for the upper red
    upper_red = np.array([180, 255, 255])  # Upper range for red
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the two masks
    mask1 = mask1 + mask2

    # Apply morphological transformations to clean up the mask
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    mask2 = cv2.bitwise_not(mask1)  # Invert the mask for the non-red parts

    # Use bitwise operations to isolate the red parts and non-red parts
    res1 = cv2.bitwise_and(img, img, mask=mask2)
    res2 = cv2.bitwise_and(background, background, mask=mask1)

    # Combine the results to replace the red parts with the background
    final = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the final image with the replaced background
    cv2.imshow("final", final)

    # Break the loop when 'c' key is pressed
    key = cv2.waitKey(1)
    if key == ord('c'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()

# End of script
