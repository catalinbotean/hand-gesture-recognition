import cv2 as cv

# Open the default camera
cap = cv.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the frame width and height
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object (try 'mp4v' for MP4)
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

if not out.isOpened():
    print("Error: Could not open video writer.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Write the frame
    out.write(frame)

    # Display the frame
    cv.imshow('Recording', frame)

    # Exit the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv.destroyAllWindows()
