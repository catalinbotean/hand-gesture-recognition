import cv2
import numpy as np

video_path = '/Users/catalinbotean/Desktop/1.mov'
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('/Users/catalinbotean/Desktop/new_video.mov', fourcc, fps, (width, height), isColor=False)

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Decrease the brightness by scaling down the pixel values
    # brightness_factor = 1.2 # Decrease brightness to 60% of the original
    # frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0.5)


    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)

    # Apply a Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use Canny edge detection to find the edges in the frame
    edges = cv2.Canny(blurred, 30, 120)

    # Apply dilation to the edges to make them thicker and more defined
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a blank image with the same dimensions as the original frame
    contour_image = np.zeros_like(frame)

    # Draw the contours on the blank image with a thicker line
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=3)

    # Convert the contour image to grayscale
    contour_gray = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image
    _, binary = cv2.threshold(contour_gray, 1, 255, cv2.THRESH_BINARY)

    # Invert the binary image to get the silhouette effect
    binary = cv2.bitwise_not(binary)

    # Optionally, further refine the silhouette
    binary = cv2.erode(binary, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=2)

    # Write the processed frame to the output video
    out.write(binary)
  # Display the resulting frame (optional, for debugging)
    cv2.imshow('Silhouette Video', binary)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
