import cv2
import numpy as np

# Global variables to store points and labels
points = []  # list to store coordinates
labels = []  # list to store labels (0 or 1)
current_label = 1  # Start with label 1 (green)

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global points, labels, image, image_copy, current_label
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the point is close to any existing point
        tolerance = 10  # Tolerance for detecting if a point was clicked
        for i, (px, py) in enumerate(points):
            if abs(px - x) < tolerance and abs(py - y) < tolerance:
                # Toggle the label of the point
                labels[i] = 0 if labels[i] == 1 else 1
                break
        else:
            # If no point was close, add a new point
            points.append([x, y])
            labels.append(current_label)
        
        # Redraw all points on the image
        redraw_image()

def redraw_image():
    global image, points, labels, image_copy
    
    # Reset the image to its original state
    image = image_copy.copy()
    
    # Draw the points with the updated labels
    for i, (px, py) in enumerate(points):
        color = (0, 255, 0) if labels[i] == 1 else (0, 0, 255)  # Green for 1, Red for 0
        cv2.circle(image, (px, py), 3, color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(labels[i]), (px + 5, py - 5), font, 0.8, color, 2)
    
    # Display the updated image
    cv2.imshow('Image', image)

def main(video_path):
    global image, image_copy, current_label

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Read the first frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    
    # Set the image as the first frame
    image = frame
    image_copy = image.copy()  # Copy of the image to restore after undoing
    
    # Create a window and set the mouse callback
    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', mouse_callback)

    while True:
        key = cv2.waitKey(0)  # Wait for a key press
        
        if key == 32:  # Space bar key (ASCII 32) to toggle between 0 and 1
            current_label = 0 if current_label == 1 else 1  # Toggle label
            print(f"Toggled label to: {current_label}")
            labels[-1] = current_label  # Update the label of the last point
            redraw_image()  # Refresh the image with the updated label
        
        elif key == 13:  # Enter key (ASCII 13) to exit
            print("Exiting program...")
            break
    
    # Output the points and labels
    print("Points and Labels:")
    print(f"Points: {points}")
    print(f"Labels: {labels}")
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "demo/data/gallery/01_dog.mp4"  # Replace with the path to your video file
    main(video_path)

# 空格切换label类型，回车退出程序
