import cv2
import gradio as gr
import numpy as np

# Global variables to store points and their labels
points = []  # list to store coordinates
labels = []  # list to store labels

# Function to handle clicks and labeling
def label_image(image, x, y):
    global points, labels

    # Toggle label between 1 and 0
    if len(labels) == 0 or labels[-1] == 1:
        label = 0
    else:
        label = 1

    # Add point and label
    points.append([x, y])
    labels.append(label)

    # Convert image to RGB (Gradio works with RGB images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw the point on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'{label}', (x + 5, y - 5), font, 0.8, (0, 0, 255), 2)
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # Return the modified image and the current points and labels
    return image, f"Points: {points}\nLabels: {labels}"

# Load image
image_path = 'test.png'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print("Error: Image not found!")
    exit()

# Convert image to RGB (Gradio works with RGB images)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Gradio Interface
iface = gr.Interface(
    fn=label_image, 
    inputs=[gr.Image(type="numpy"), gr.Slider(minimum=0, maximum=image.shape[1]-1, step=1, label="X position"), gr.Slider(minimum=0, maximum=image.shape[0]-1, step=1, label="Y position")],
    outputs=[gr.Image(type="numpy"), gr.Textbox()],
    live=True
)

iface.launch()
